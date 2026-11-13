"""Generate combined INT8 model and validate the four final ONNX models.
Run from the folder containing project.py: python finalize_edge_models.py
Optional final held-out test: python finalize_edge_models.py --test
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import project as core


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default=str(core.ROOT / 'data' / 'trashnet'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--calibration-images', type=int, default=150)
    ap.add_argument('--test', action='store_true', help='Use held-out test set only for final evaluation')
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        ap.error(f'Dataset not found: {data_dir}')
    cfg = core.OUT / 'config.json'
    if cfg.exists():
        saved = json.loads(cfg.read_text())
        if saved.get('seed') != args.seed or Path(saved.get('data_dir', '')).resolve() != data_dir.resolve():
            ap.error('Seed or data folder differs from original training configuration.')

    baseline = core.OUT / 'extended' / 'baseline.onnx'
    if not baseline.exists():
        baseline = core.OUT / 'baseline.onnx'
    int8 = core.OUT / 'extended' / 'static_int8.onnx'
    if not int8.exists():
        int8 = core.OUT / 'int8.onnx'
    selective = core.OUT / 'selective' / 'selective_structured.onnx'
    combined = core.OUT / 'selective' / 'selective_structured_static_int8.onnx'
    for name, path in [('baseline', baseline), ('static INT8', int8), ('six-group selective model', selective)]:
        if not path.exists():
            ap.error(f'Missing {name}: {path}. Copy the six-group model into outputs/selective/ if needed.')
    splits = core.split_samples(data_dir, args.seed)
    if not combined.exists():
        print('Quantizing six-group structured model using training-only calibration images...', flush=True)
        core.static_int8(selective, combined, splits['train'], args.calibration_images)
        print('Saved:', combined)
    else:
        print('Combined model already exists; keeping:', combined)

    split = 'test' if args.test else 'val'
    loader = core.loaders(SimpleNamespace(data_dir=str(data_dir), seed=args.seed, batch=32))[split]
    models = [('baseline', baseline), ('static_int8', int8), ('selective_structured_6groups', selective), ('selective_structured_6groups_int8', combined)]
    rows = []
    for label, path in models:
        session = core.session(path)
        truth, predictions = core.evaluate_predictions(session, loader)
        accuracy = float((truth == predictions).mean())
        rows.append({'model': label, 'split': split, 'accuracy': accuracy, 'accuracy_pct': accuracy * 100, 'size_mb': path.stat().st_size / 1048576, 'file': str(path)})
        print(f'{label:36} {accuracy*100:7.2f}%  {path.stat().st_size/1048576:7.3f} MB')
    output = core.OUT / 'selective' / f'final_model_{split}_comparison.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', output)
    print('Note: latency and RAM should be benchmarked on the Surface Pro 7, not inferred from this CSV.')

if __name__ == '__main__':
    main()
