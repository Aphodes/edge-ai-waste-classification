"""TrashNet MobileNetV2 compression experiment. Run `python project.py --help`."""
import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import torch
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
SIZE = 224
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def preprocess(training=False):
    ops = [transforms.Resize((SIZE, SIZE))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class IndexedImages(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = self.transform(img.convert('RGB'))
        return image, label


def split_samples(data_dir, seed):
    folder = datasets.ImageFolder(str(data_dir))
    if folder.classes != CLASSES:
        raise ValueError(f'Expected class folders {CLASSES}; found {folder.classes}')
    rng = np.random.default_rng(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label, cls in enumerate(CLASSES):
        samples = [sample for sample in folder.samples if sample[1] == label]
        rng.shuffle(samples)
        n = len(samples)
        if n < 7:
            raise ValueError(f'Class {cls} needs at least 7 images for the requested split.')
        n_train = int(round(n * .70))
        n_val = int(round(n * .15))
        splits['train'].extend(samples[:n_train])
        splits['val'].extend(samples[n_train:n_train + n_val])
        splits['test'].extend(samples[n_train + n_val:])
    for items in splits.values():
        rng.shuffle(items)
    return splits


def loaders(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'TrashNet not found at {data_dir}. See README.md for folder layout.')
    splits = split_samples(data_dir, args.seed)
    return {
        name: DataLoader(IndexedImages(items, preprocess(name == 'train')),
                         batch_size=args.batch if name == 'train' else 1,
                         shuffle=(name == 'train'), num_workers=0)
        for name, items in splits.items()
    }


def make_model(pretrained=False):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_torch(m, loader, device):
    m.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = m(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / total


def train(args):
    set_seed(args.seed)
    data = loaders(args)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    m = make_model(pretrained=True).to(device)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.features[-2:].parameters():
        p.requires_grad = True
    for p in m.classifier.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val = -1
    history = []
    for epoch in range(args.epochs):
        m.train()
        # Frozen layers' batch-normalization running statistics should stay fixed.
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        loss_sum = 0
        for x, y in data['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(m(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        val_acc = accuracy_torch(m, data['val'], device)
        train_loss = loss_sum / len(data['train'].dataset)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_accuracy': val_acc})
        print(f'epoch {epoch + 1}/{args.epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}', flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(m.cpu().state_dict(), OUT / 'baseline.pt')
            m.to(device)
    pd.DataFrame(history).to_csv(OUT / 'training_history.csv', index=False)
    (OUT / 'config.json').write_text(json.dumps({'classes': CLASSES, 'image_size': SIZE,
                                                   'seed': args.seed, 'data_dir': str(Path(args.data_dir).resolve()),
                                                   'split': [0.70, 0.15, 0.15]}, indent=2))
    print(f'Best validation accuracy: {best_val:.4f}. Saved {OUT / "baseline.pt"}')


def load_model():
    path = OUT / 'baseline.pt'
    if not path.exists():
        raise FileNotFoundError('Run train first.')
    m = make_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return m.eval()


def prune(m, amount):
    # Unstructured sparsity, NOT structured channel removal. Dense ONNX may not speed up.
    import torch.nn.utils.prune as pruning
    for layer in m.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pruning.l1_unstructured(layer, name='weight', amount=amount)
            pruning.remove(layer, 'weight')
    return m.eval()


def export(m, path):
    m.eval().cpu()
    with torch.inference_mode():
        torch.onnx.export(m, torch.randn(1, 3, SIZE, SIZE), str(path),
                          input_names=['input'], output_names=['logits'], opset_version=17,
                          dynamo=False)


class CalibrationReader(CalibrationDataReader):
    def __init__(self, samples, max_images):
        self.samples = samples[:max_images]
        self.transform = preprocess(False)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        path, _ = self.samples[self.index]
        self.index += 1
        with Image.open(path) as img:
            x = self.transform(img.convert('RGB')).unsqueeze(0).numpy()
        return {'input': x}

    def rewind(self):
        self.index = 0


def static_int8(src, dst, samples, calibration_images):
    quantize_static(str(src), str(dst), CalibrationReader(samples, calibration_images),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax)


def build(args):
    data = split_samples(Path(args.data_dir), args.seed)
    base = load_model()
    export(base, OUT / 'baseline.onnx')
    print('Calibrating baseline INT8 model...', flush=True)
    static_int8(OUT / 'baseline.onnx', OUT / 'int8.onnx', data['train'], args.calibration_images)
    for amount in args.pruning_levels:
        if not 0 < amount < 1:
            raise ValueError('Pruning levels must be fractions between 0 and 1, e.g. 0.2')
        label = int(round(amount * 100))
        m = prune(copy.deepcopy(base), amount)
        fp = OUT / f'pruned_{label}.onnx'
        export(m, fp)
        print(f'Calibrating combined_{label}...', flush=True)
        static_int8(fp, OUT / f'combined_{label}.onnx', data['train'], args.calibration_images)
    print('Model exports saved in', OUT)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def evaluate_predictions(sess, loader):
    input_name = sess.get_inputs()[0].name
    y_true, y_pred = [], []
    for x, y in loader:
        logits = sess.run(None, {input_name: x.numpy()})[0]
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(logits, axis=1).tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def benchmark(sess, repeats=150, warmup=25):
    input_name = sess.get_inputs()[0].name
    x = np.random.default_rng(42).normal(size=(1, 3, SIZE, SIZE)).astype('float32')
    for _ in range(warmup):
        sess.run(None, {input_name: x})
    proc = psutil.Process()
    start_cpu = proc.cpu_times()
    start_wall = time.perf_counter()
    latencies = []
    peak_rss = proc.memory_info().rss
    for _ in range(repeats):
        t = time.perf_counter()
        sess.run(None, {input_name: x})
        latencies.append((time.perf_counter() - t) * 1000)
        peak_rss = max(peak_rss, proc.memory_info().rss)
    elapsed = time.perf_counter() - start_wall
    end_cpu = proc.cpu_times()
    cpu_seconds = end_cpu.user + end_cpu.system - start_cpu.user - start_cpu.system
    return {'latency_median_ms': float(np.median(latencies)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'throughput_fps': float(1000 / np.mean(latencies)),
            'process_rss_mb': float(peak_rss / 1048576),
            'cpu_one_core_percent': float(100 * cpu_seconds / elapsed)}


def evaluate(args):
    config_path = OUT / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError('Run train first.')
    config = json.loads(config_path.read_text())
    if args.seed != config['seed'] or Path(args.data_dir).resolve() != Path(config['data_dir']):
        raise ValueError('Use the same --seed and --data-dir as training for a consistent test split.')
    test = loaders(args)['test']
    paths = [OUT / 'baseline.onnx', OUT / 'int8.onnx']
    paths += sorted(OUT.glob('pruned_*.onnx')) + sorted(OUT.glob('combined_*.onnx'))
    if not all(p.exists() for p in paths[:2]):
        raise FileNotFoundError('Run build first.')
    rows = []
    for path in paths:
        print('Evaluating', path.name, flush=True)
        sess = session(path)
        truth, predictions = evaluate_predictions(sess, test)
        metrics = [benchmark(sess, args.repeats) for _ in range(args.trials)]
        row = {'model': path.stem, 'accuracy': float(np.mean(truth == predictions)),
               'size_mb': path.stat().st_size / 1048576}
        for key in metrics[0]:
            values = [trial[key] for trial in metrics]
            row[key] = float(np.mean(values))
            row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for index, cls in enumerate(CLASSES):
            tp = int(np.sum((truth == index) & (predictions == index)))
            fp = int(np.sum((truth != index) & (predictions == index)))
            fn = int(np.sum((truth == index) & (predictions != index)))
            row[f'{cls}_recall'] = tp / (tp + fn) if tp + fn else 0
            row[f'{cls}_precision'] = tp / (tp + fp) if tp + fp else 0
            row[f'{cls}_f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        for actual, predicted in zip(truth, predictions):
            confusion[actual, predicted] += 1
        pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(OUT / f'confusion_{path.stem}.csv')
        rows.append(row)
        del sess
        gc.collect()
    df = pd.DataFrame(rows)
    base = df.loc[df.model == 'baseline'].iloc[0]
    df['accuracy_drop_pp'] = (base.accuracy - df.accuracy) * 100
    df['size_reduction_x'] = base.size_mb / df.size_mb
    df['speedup_x'] = base.latency_median_ms / df.latency_median_ms
    df['within_2pp_accuracy'] = df.accuracy_drop_pp <= 2
    df.to_csv(OUT / 'results.csv', index=False)
    print(df[['model', 'accuracy', 'size_mb', 'latency_median_ms', 'speedup_x', 'accuracy_drop_pp']].to_string(index=False))
    plot(df)


def plot(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.model, df.latency_median_ms)
    ax.set_ylabel('Median latency (ms)')
    ax.tick_params(axis='x', rotation=50)
    fig.tight_layout()
    fig.savefig(OUT / 'latency.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df.size_mb, df.accuracy * 100)
    for _, row in df.iterrows():
        ax.annotate(row.model, (row.size_mb, row.accuracy * 100), fontsize=7)
    ax.set_xlabel('ONNX model size (MB)')
    ax.set_ylabel('Test accuracy (%)')
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy_size.png', dpi=180)
    plt.close(fig)
    sub = df[df.model.str.startswith('pruned_')].copy()
    if not sub.empty:
        sub['pruning_percent'] = sub.model.str.split('_').str[-1].astype(int)
        sub = sub.sort_values('pruning_percent')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub.pruning_percent, sub.accuracy * 100, marker='o')
        ax.set_xlabel('Unstructured pruning (%)')
        ax.set_ylabel('Test accuracy (%)')
        fig.tight_layout()
        fig.savefig(OUT / 'pruning_sweep.png', dpi=180)
        plt.close(fig)


def camera(args):
    import cv2
    path = OUT / f'{args.variant}.onnx'
    if not path.exists():
        raise FileNotFoundError(path)
    sess = session(path)
    name = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError('Camera unavailable')
    print('Press q to quit; show one waste item at a time.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = preprocess(False)(Image.fromarray(rgb)).unsqueeze(0).numpy()
            start = time.perf_counter()
            logits = sess.run(None, {name: x})[0][0]
            latency = (time.perf_counter() - start) * 1000
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            index = int(np.argmax(probs))
            cv2.putText(frame, f'{CLASSES[index]} {probs[index]:.1%} | {latency:.1f} ms',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
            cv2.imshow('TrashNet Edge AI - press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'build', 'evaluate', 'camera'])
    parser.add_argument('--data-dir', default=str(ROOT / 'data' / 'trashnet'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibration-images', type=int, default=150)
    parser.add_argument('--pruning-levels', type=float, nargs='+', default=[.2, .4, .6, .8])
    parser.add_argument('--repeats', type=int, default=150)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--variant', default='int8')
    parser.add_argument('--camera-index', type=int, default=0)
    args = parser.parse_args()
    {'train': train, 'build': build, 'evaluate': evaluate, 'camera': camera}[args.command](args)


if __name__ == '__main__':
    main()
