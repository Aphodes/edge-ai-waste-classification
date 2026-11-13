"""Additional TrashNet compression experiments. Run: python experiments.py --help"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import prune as pruning

import project as core

EXP = core.OUT / 'extended'
EXP.mkdir(parents=True, exist_ok=True)


def validate_config(args):
    cfg = core.OUT / 'config.json'
    if not cfg.exists():
        raise FileNotFoundError('Missing outputs/config.json; use the original project training configuration.')
    saved = json.loads(cfg.read_text())
    if saved['seed'] != args.seed or Path(saved['data_dir']).resolve() != Path(args.data_dir).resolve():
        raise ValueError('Seed and data directory must match the original baseline training.')


def fine_tune(model, args, label, masked=False):
    data = core.loaders(args)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = model.to(device)
    # Train all remaining weights after pruning; preserve BN running statistics.
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr)
    loss_fn = nn.CrossEntropyLoss()
    best = -1.0
    history = []
    best_state = None
    for epoch in range(args.finetune_epochs):
        model.train()
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        loss_total = 0.0
        for x, y in data['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_total += float(loss.item()) * len(y)
        val = core.accuracy_torch(model, data['val'], device)
        history.append({'epoch': epoch+1, 'train_loss': loss_total / len(data['train'].dataset), 'val_accuracy': val})
        print(f'{label}: epoch {epoch+1}/{args.finetune_epochs} validation={val:.4f}', flush=True)
        if val > best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(EXP / f'{label}_finetune_history.csv', index=False)
    model.cpu().eval()
    return model


def unstructured(base, ratio, args, label):
    model = copy.deepcopy(base)
    modules = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            pruning.l1_unstructured(mod, name='weight', amount=ratio)
            modules.append(mod)
    # Keep masks during fine-tuning, so pruned weights cannot regrow.
    if args.finetune_epochs:
        model = fine_tune(model, args, label)
    for mod in modules:
        pruning.remove(mod, 'weight')
    return model.eval()


def structured(base, ratio, args, label):
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError('Structured pruning needs torch-pruning: pip install torch-pruning') from exc
    model = copy.deepcopy(base).eval()
    # Dependency-aware channel pruning handles MobileNetV2 skip connections and depthwise layers.
    # Exclude the classification head and final projection from pruning.
    ignored = [model.classifier[1], model.features[-1][0]]
    example = torch.randn(1, 3, core.SIZE, core.SIZE)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example, importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=ratio, iterative_steps=1, ignored_layers=ignored,
    )
    pruner.step()
    with torch.no_grad():
        result = model(example)
    if result.shape != (1, len(core.CLASSES)):
        raise RuntimeError(f'Unexpected structured model output shape: {result.shape}')
    if args.finetune_epochs:
        model = fine_tune(model, args, label)
    return model.eval()


def dynamic_quant(src, dst):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def build(args):
    validate_config(args)
    core.set_seed(args.seed)
    splits = core.split_samples(Path(args.data_dir), args.seed)
    base = core.load_model()
    core.export(base, EXP / 'baseline.onnx')
    core.static_int8(EXP / 'baseline.onnx', EXP / 'static_int8.onnx', splits['train'], args.calibration_images)
    if args.dynamic:
        dynamic_quant(EXP / 'baseline.onnx', EXP / 'dynamic_int8.onnx')
    for kind, levels in [('unstructured', args.unstructured), ('structured', args.structured)]:
        for ratio in levels:
            if not 0 < ratio < 1:
                raise ValueError(f'Invalid pruning fraction: {ratio}')
            label = f'{kind}_{int(round(100*ratio))}'
            print('Building', label, flush=True)
            m = unstructured(base, ratio, args, label) if kind == 'unstructured' else structured(base, ratio, args, label)
            fp = EXP / f'{label}.onnx'
            core.export(m, fp)
            core.static_int8(fp, EXP / f'{label}_static_int8.onnx', splits['train'], args.calibration_images)
            if args.dynamic:
                dynamic_quant(fp, EXP / f'{label}_dynamic_int8.onnx')
            print('Saved', label, flush=True)
    print('Finished. Models:', EXP)


def evaluate(args):
    validate_config(args)
    test = core.loaders(args)['test']
    paths = sorted(EXP.glob('*.onnx'))
    if not paths:
        raise FileNotFoundError('No models in outputs/extended; run build first.')
    rows = []
    for path in paths:
        print('Evaluating', path.name, flush=True)
        sess = core.session(path)
        truth, pred = core.evaluate_predictions(sess, test)
        trials = [core.benchmark(sess, args.repeats) for _ in range(args.trials)]
        row = {'model': path.stem, 'accuracy': float(np.mean(truth == pred)), 'size_mb': path.stat().st_size / 1048576}
        for key in trials[0]:
            vals = [trial[key] for trial in trials]
            row[key] = float(np.mean(vals))
            row[key + '_std'] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
        for index, cls in enumerate(core.CLASSES):
            tp = int(np.sum((truth==index)&(pred==index)))
            fp = int(np.sum((truth!=index)&(pred==index)))
            fn = int(np.sum((truth==index)&(pred!=index)))
            row[f'{cls}_precision'] = tp/(tp+fp) if tp+fp else 0.0
            row[f'{cls}_recall'] = tp/(tp+fn) if tp+fn else 0.0
            row[f'{cls}_f1'] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        pd.crosstab(pd.Series(truth,name='actual'), pd.Series(pred,name='predicted')).reindex(
            index=range(6),columns=range(6),fill_value=0).set_axis(core.CLASSES,axis=0).set_axis(core.CLASSES,axis=1).to_csv(EXP/f'confusion_{path.stem}.csv')
        rows.append(row)
        del sess
    df = pd.DataFrame(rows)
    base = df.loc[df.model=='baseline'].iloc[0]
    df['accuracy_drop_pp'] = (base.accuracy-df.accuracy)*100
    df['size_reduction_x'] = base.size_mb/df.size_mb
    df['speedup_x'] = base.latency_median_ms/df.latency_median_ms
    df['within_2pp_accuracy'] = df.accuracy_drop_pp <= 2
    df.to_csv(EXP/'results_extended.csv',index=False)
    print(df[['model','accuracy','size_mb','latency_median_ms','accuracy_drop_pp','speedup_x']].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['build','evaluate'])
    ap.add_argument('--data-dir',default=str(core.ROOT/'data'/'trashnet'))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--unstructured',type=float,nargs='*',default=[.05,.10,.15,.20,.40])
    ap.add_argument('--structured',type=float,nargs='*',default=[.10,.20,.30])
    ap.add_argument('--finetune-epochs',type=int,default=3)
    ap.add_argument('--finetune-lr',type=float,default=1e-5)
    ap.add_argument('--calibration-images',type=int,default=150)
    ap.add_argument('--dynamic',action='store_true',help='Also export dynamic INT8 variants; these may not accelerate convolutions')
    ap.add_argument('--trials',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=150)
    args=ap.parse_args()
    {'build':build,'evaluate':evaluate}[args.command](args)

if __name__=='__main__':
    main()
