"""Fine-tuning-only control for TrashNet MobileNetV2; does not overwrite baseline.
Run beside project.py and experiments.py:
  python finetune_control.py --epochs 5
"""
import argparse
import json
from pathlib import Path
import torch
import project as core
import experiments


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--finetune-lr', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--data-dir', default=str(core.ROOT / 'data' / 'trashnet'))
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error('--epochs must be at least 1')
    config = json.loads((core.OUT / 'config.json').read_text())
    if args.seed != config['seed'] or Path(args.data_dir).resolve() != Path(config['data_dir']).resolve():
        ap.error('Seed/data-dir must match original training configuration')
    args.finetune_epochs = args.epochs
    core.set_seed(args.seed)
    output = core.OUT / 'control'
    output.mkdir(parents=True, exist_ok=True)
    model = core.load_model()
    data = core.loaders(args)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    before = core.accuracy_torch(model.to(device), data['val'], device)
    model.cpu()
    print(f'Baseline validation accuracy: {before:.4%}', flush=True)
    model = experiments.fine_tune(model, args, 'finetune_only_control')
    after = core.accuracy_torch(model.to(device), data['val'], device)
    model.cpu().eval()
    torch.save(model.state_dict(), output / 'finetune_only_control.pt')
    core.export(model, output / 'finetune_only_control.onnx')
    src = core.OUT / 'extended' / 'finetune_only_control_finetune_history.csv'
    if src.exists():
        import shutil
        shutil.copy2(src, output / 'finetune_only_control_history.csv')
    summary = {'baseline_val_accuracy': before, 'control_val_accuracy': after,
               'additional_epochs': args.epochs, 'learning_rate': args.finetune_lr,
               'test_set_used': False}
    (output / 'control_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Saved control model and history to', output)

if __name__ == '__main__':
    main()
