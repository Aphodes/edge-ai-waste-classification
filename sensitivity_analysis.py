"""TrashNet MobileNetV2: sensitivity scan and conservative selective structured pruning.
Place alongside project.py and experiments.py. See SELECTIVE_PRUNING_README.md.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import project as core

OUT = core.OUT / 'selective'
OUT.mkdir(parents=True, exist_ok=True)


def dependencies():
    try:
        import torch_pruning as tp
    except ImportError as e:
        raise RuntimeError('Install torch-pruning in your active environment: python -m pip install torch-pruning') from e
    return tp


def candidates(model):
    """Only internal expansion convs; avoid block output widths and skip-add dimensions."""
    result = []
    for i, block in enumerate(model.features):
        if not isinstance(block, nn.Module) or not hasattr(block, 'conv'):
            continue
        first = block.conv[0]
        if isinstance(first, nn.Sequential) and isinstance(first[0], nn.Conv2d):
            conv = first[0]
            if conv.kernel_size == (1, 1) and conv.out_channels > 8:
                result.append((f'features.{i}.conv.0.0', conv))
    return result


def channel_indices(conv, fraction, min_remaining=8):
    n = conv.out_channels
    remove = min(max(1, int(round(n * fraction))), n - min_remaining)
    if remove <= 0:
        raise ValueError('No removable channels in this layer')
    scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.argsort(scores)[:remove].tolist()


def apply_one(model, layer_name, fraction):
    tp = dependencies()
    lookup = dict(model.named_modules())
    if layer_name not in lookup:
        raise ValueError(f'Layer {layer_name} not found in model')
    conv = lookup[layer_name]
    before = conv.out_channels
    idxs = channel_indices(conv, fraction)
    example = torch.randn(1, 3, core.SIZE, core.SIZE)
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    group = dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
    if not dg.check_pruning_group(group):
        raise RuntimeError(f'Invalid pruning dependency group for {layer_name}')
    group.prune()
    with torch.no_grad():
        output = model(example)
    if tuple(output.shape) != (1, len(core.CLASSES)):
        raise RuntimeError(f'Invalid output after pruning: {tuple(output.shape)}')
    return before, dict(model.named_modules())[layer_name].out_channels


def complexity(model):
    tp = dependencies()
    try:
        with torch.no_grad():
            macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, core.SIZE, core.SIZE))
        return float(macs), int(params)
    except Exception as e:
        raise RuntimeError(f'Cannot measure MACs/parameters: {e}') from e


def accuracy(model, loader, device):
    model.to(device)
    value = core.accuracy_torch(model, loader, device)
    model.cpu().eval()
    return float(value)


def common(args):
    cfg = core.OUT / 'config.json'
    if not cfg.exists():
        raise FileNotFoundError('Missing outputs/config.json from original TrashNet training')
    saved = json.loads(cfg.read_text())
    if saved['seed'] != args.seed or Path(saved['data_dir']).resolve() != Path(args.data_dir).resolve():
        raise ValueError('Seed/data-dir differ from baseline training. Use original split configuration.')
    core.set_seed(args.seed)
    data = core.loaders(args)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    return data, device


def scan(args):
    data, device = common(args)
    base = core.load_model().cpu().eval()
    base_accuracy = accuracy(base, data['val'], device)
    base_macs, base_params = complexity(base)
    rows = []
    targets = [name for name, _ in candidates(base)]
    if not targets:
        raise RuntimeError('No expansion convolution candidates found; check torchvision MobileNetV2 version.')
    print(f'Baseline validation accuracy: {base_accuracy:.4%}; {len(targets)} candidate layers', flush=True)
    for name in targets:
        for fraction in args.fractions:
            model = copy.deepcopy(base).eval()
            try:
                original, remaining = apply_one(model, name, fraction)
                val = accuracy(model, data['val'], device)
                macs, params = complexity(model)
                row = dict(layer=name, requested_fraction=fraction, channels_before=original,
                           channels_after=remaining, channels_removed=original-remaining,
                           actual_fraction=(original-remaining)/original,
                           baseline_val_accuracy=base_accuracy, pruned_val_accuracy=val,
                           accuracy_drop_pp=100*(base_accuracy-val),
                           macs_before=base_macs, macs_after=macs,
                           macs_saved_pct=100*(base_macs-macs)/base_macs,
                           parameters_before=base_params, parameters_after=params,
                           parameters_saved_pct=100*(base_params-params)/base_params,
                           status='ok', error='')
                print(f'{name} {fraction:.0%}: accuracy drop {row["accuracy_drop_pp"]:+.2f} pp; MACs saved {row["macs_saved_pct"]:.2f}%', flush=True)
            except Exception as e:
                row = dict(layer=name, requested_fraction=fraction, status='error', error=str(e))
                print(f'FAILED {name} {fraction:.0%}: {e}', flush=True)
            rows.append(row)
            write_csv(OUT / 'sensitivity_results.csv', rows)
    print('Saved', OUT / 'sensitivity_results.csv')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select(args):
    data, device = common(args)
    scan_path = OUT / 'sensitivity_results.csv'
    if not scan_path.exists():
        raise FileNotFoundError('Run scan first')
    with scan_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    choices = []
    for r in rows:
        if r['status'] != 'ok':
            continue
        loss = float(r['accuracy_drop_pp'])
        saved = float(r['macs_saved_pct'])
        if saved <= 0 or loss > args.max_single_drop_pp or float(r['requested_fraction']) != args.fraction:
            continue
        # Negative loss (accuracy gain) gets no artificial infinite score.
        choices.append((saved / max(0.25, loss), r['layer'], r))
    choices.sort(reverse=True)
    model = core.load_model().cpu().eval()
    base_acc = accuracy(model, data['val'], device)
    base_macs, base_params = complexity(model)
    accepted = []
    for _, name, _ in choices:
        if len(accepted) >= args.max_groups:
            break
        trial = copy.deepcopy(model).eval()
        try:
            original, remaining = apply_one(trial, name, args.fraction)
            trial_acc = accuracy(trial, data['val'], device)
            total_drop = (base_acc-trial_acc)*100
            trial_macs, _ = complexity(trial)
            total_saved = 100*(base_macs-trial_macs)/base_macs
            if total_drop <= args.max_total_drop_pp and total_saved > 0:
                model = trial
                accepted.append(dict(layer=name, before=original, after=remaining,
                                     validation_accuracy=trial_acc, cumulative_accuracy_drop_pp=total_drop,
                                     cumulative_macs_saved_pct=total_saved))
                print(f'ACCEPT {name}: cumulative drop {total_drop:.2f} pp, MACs saved {total_saved:.2f}%', flush=True)
            else:
                print(f'SKIP {name}: cumulative drop {total_drop:.2f} pp, MACs saved {total_saved:.2f}%', flush=True)
        except Exception as e:
            print(f'SKIP {name}: {e}', flush=True)
    if not accepted:
        raise RuntimeError('No candidates met constraints. Try --max-total-drop-pp 2.0 or --fraction 0.05 after scanning.')
    write_csv(OUT / 'selected_groups.csv', accepted)
    pre_acc = accuracy(model, data['val'], device)
    print(f'Before fine-tuning: validation accuracy {pre_acc:.4%}', flush=True)
    if args.finetune_epochs > 0:
        # Reuse the existing fine-tuning function; it uses train/validation only.
        import experiments
        model = experiments.fine_tune(model, args, 'selective_structured', masked=False)
        history = core.OUT / 'extended' / 'selective_structured_finetune_history.csv'
        if history.exists():
            import shutil
            shutil.copy2(history, OUT / 'selective_structured_finetune_history.csv')
    post_acc = accuracy(model, data['val'], device)
    final_macs, final_params = complexity(model)
    torch.save(model.state_dict(), OUT / 'selective_structured_state.pt')
    fp = OUT / 'selective_structured.onnx'
    core.export(model, fp)
    summary = dict(baseline_validation_accuracy=base_acc, before_finetune_validation_accuracy=pre_acc,
                   after_finetune_validation_accuracy=post_acc, groups_pruned=len(accepted),
                   macs_saved_pct=100*(base_macs-final_macs)/base_macs,
                   parameters_saved_pct=100*(base_params-final_params)/base_params,
                   onnx_file_mb=fp.stat().st_size/1048576)
    (OUT / 'selection_summary.json').write_text(json.dumps(summary, indent=2))
    if args.quantize:
        samples = core.split_samples(Path(args.data_dir), args.seed)['train']
        core.static_int8(fp, OUT / 'selective_structured_static_int8.onnx', samples, args.calibration_images)
    print('Saved', fp, 'and selection_summary.json')
    print(json.dumps(summary, indent=2))
    print('Final test-set evaluation: use the existing evaluation workflow only after selecting the final model.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['scan', 'select'])
    ap.add_argument('--data-dir', default=str(core.ROOT/'data'/'trashnet'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--fractions', type=float, nargs='+', default=[.05, .10], help='Scan fractions per eligible layer')
    ap.add_argument('--fraction', type=float, default=.05, help='Selection uses scan results for this exact fraction')
    ap.add_argument('--max-single-drop-pp', type=float, default=1.0)
    ap.add_argument('--max-total-drop-pp', type=float, default=1.5)
    ap.add_argument('--max-groups', type=int, default=3)
    ap.add_argument('--finetune-epochs', type=int, default=5)
    ap.add_argument('--finetune-lr', type=float, default=1e-5)
    ap.add_argument('--quantize', action='store_true')
    ap.add_argument('--calibration-images', type=int, default=150)
    args = ap.parse_args()
    if args.command == 'scan':
        scan(args)
    else:
        select(args)


if __name__ == '__main__':
    main()
