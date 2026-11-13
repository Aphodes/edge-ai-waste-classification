# Edge AI Waste Classification: MobileNetV2 Compression and Deployment

ESP3201A project exploring the trade-offs among **classification accuracy, model size, and inference efficiency** when applying pruning and INT8 quantization to a six-class waste classifier. The project uses **TrashNet**, **MobileNetV2**, **PyTorch**, and **ONNX Runtime**. A separate Windows camera application was developed for benchmarking on a Surface Pro 7.

## Research question

Can pruning and INT8 post-training quantization reduce model size and inference latency without an unacceptable loss in waste-classification accuracy? We used an initial target of **at most 2 percentage points of test-accuracy loss** relative to the FP32 baseline.

## Dataset and architecture

- Dataset: [TrashNet](https://github.com/garythung/trashnet), 2,527 images in six classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Split: deterministic, class-stratified **70% train / 15% validation / 15% test**, seed **42**.
- Input: **224 x 224 RGB**, normalized with ImageNet mean and standard deviation; training uses horizontal flips and small rotations.
- Model: ImageNet-pretrained **MobileNetV2** with a six-class classification head. Initial baseline training fine-tunes the classifier and final feature blocks for **8 epochs**.
- Deployment: exported **ONNX** models executed locally with ONNX Runtime; no PyTorch installation is required to run the ONNX inference app.

The classifier predicts **one category for an image**; it does **not** localize multiple objects or guarantee that a material is accepted by a local recycling program.

## Code layout

Keep the research scripts together in the repository root because they import `project` and `experiments` by module name:

```text
edge-ai-waste-classification/
  project.py
  experiments.py
  sensitivity_analysis.py
  finetune_control.py
  finalize_edge_models.py
  requirements.txt
  README.md
  .gitignore
  models/                   # curated final ONNX exports (optional for training)
  results/                  # curated experimental CSV/JSON summaries (recommended)
  surface_app/              # add the separate Surface Pro app when available
  data/trashnet/            # downloaded locally, ignored by Git
  outputs/                  # generated locally, ignored by Git
```

### Research scripts

| Script | Role |
|---|---|
| `project.py` | Train baseline; generate initial unstructured-pruned and static-INT8 ONNX variants; evaluate models. |
| `experiments.py` | Extended unstructured/structured pruning, post-pruning fine-tuning, static INT8, test metrics and confusion matrices. |
| `sensitivity_analysis.py` | Scan MobileNetV2 expansion layers and selectively prune channels subject to validation-accuracy constraints. |
| `finetune_control.py` | Fine-tune the unpruned baseline to separate training effects from pruning effects. |
| `finalize_edge_models.py` | Quantize the selected structured model and compare four final models on validation or held-out test images. |

The original `project.py` includes a simple OpenCV webcam command. The separate Surface Pro application adds a model selector and dedicated benchmarking.

## Setup

Use Python 3.10-3.12 where possible. On Windows PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For hardware-specific PyTorch builds, consult [PyTorch installation instructions](https://pytorch.org/get-started/locally/). The dependency file specifies compatible ranges rather than the exact versions originally used; for strict reproduction, record your tested environment with `python -m pip freeze`.

Download the TrashNet dataset separately, following its source repository's license and usage terms. Place the **six class directories directly** inside `data/trashnet/`:

```text
data/trashnet/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Do not commit the dataset or your virtual environment.

## Reproducing the experiments

The commands below are run from the repository root with the environment active. They are ordered to match the original workflow.

### 1. Train the FP32 baseline

```powershell
python project.py train --epochs 8
```

This saves `outputs/baseline.pt`, `outputs/config.json`, and `outputs/training_history.csv`. **Later research scripts require the checkpoint and configuration.** The checkpoint is ignored by Git by default; users reproducing experiments must train it or obtain it separately.

### 2. Initial unstructured pruning and INT8 comparison

```powershell
python project.py build --calibration-images 150
python project.py evaluate --trials 3 --repeats 150
```

These initial pruning sweeps were **not fine-tuned**. Their dense ONNX exports do not necessarily shrink or speed up despite having zero-valued weights.

### 3. Extended experiments with fine-tuning

```powershell
python experiments.py build --unstructured 0.05 0.10 --structured 0.10 --finetune-epochs 3
python experiments.py evaluate --trials 3 --repeats 150
```

Outputs are written under `outputs/extended/`. This stage uses `torch-pruning` for dependency-aware structured pruning and ONNX Runtime static INT8 post-training quantization. The default quantization calibration uses 150 **training** images.

### 4. Layer sensitivity and selective structured pruning

```powershell
python sensitivity_analysis.py scan --fractions 0.05 0.10
python sensitivity_analysis.py select --fraction 0.05 --max-single-drop-pp 1.0 --max-total-drop-pp 2.0 --max-groups 6 --finetune-epochs 5
```

The scan tests eligible MobileNetV2 expansion layers **individually**, recording validation-accuracy loss, parameter reduction, and MAC reduction. Selection ranks candidates and sequentially accepts pruning groups only if the combined model stays within its configured validation-accuracy-loss constraint. It then fine-tunes and exports an ONNX model.

**Important:** `--max-groups 6` means **up to** six accepted pruning groups, not necessarily six. The script writes to fixed filenames in `outputs/selective/`; back up that folder between trials. The `select` command expects `outputs/selective/sensitivity_results.csv` from a scan at the same `--fraction`.

### 5. Fine-tuning-only control

```powershell
python finetune_control.py --epochs 5
```

This evaluates the effect of extra training without pruning. It does not overwrite the baseline checkpoint.

### 6. Final structured + INT8 comparison

```powershell
python finalize_edge_models.py
python finalize_edge_models.py --test
```

This script expects `outputs/selective/selective_structured.onnx` to be the **selected six-group model**, plus baseline and INT8 models in the expected `outputs/` or `outputs/extended/` paths. It saves `selective_structured_static_int8.onnx` and validation/test comparison CSVs under `outputs/selective/`.

The four-model finalization script does **not** include `unstructured_5_static_int8.onnx`; that candidate comes from the extended experiments.

## Five deployment candidates

For the separate Surface Pro application, use the **exact filenames expected by its `app.py`**:

| Model | App filename | Held-out test accuracy | ONNX size |
|---|---|---:|---:|
| FP32 baseline | `baseline.onnx` | 90.77% | 8.487 MB |
| Static INT8 | `static_int8.onnx` | 83.91% | 2.494 MB |
| Unstructured 5% + fine-tuning + INT8 | `unstructured_5_static_int8.onnx` | 87.60% | 2.494 MB |
| Selective structured (6 groups) | `selective_structured_6groups.onnx` | 93.40% | 8.446 MB |
| Selective structured (6 groups) + INT8 | `selective_structured_6groups_static_int8.onnx` | 86.81% | 2.482 MB |

The two selective files may originally be named `selective_structured.onnx` and `selective_structured_static_int8.onnx` in the training outputs. **Copy and rename** them to the `_6groups` filenames above for the app, after verifying they are the six-group exports. The 5% unstructured **FP32** file (`unstructured_5.onnx`, approximately 8.49 MB) is *not* the quantized candidate.

Store curated files in `models/` for GitHub; the app reads its own `surface_app/models/` folder unless you change its code or copy the files there.

## Surface Pro 7 CPU benchmarks

The following values come from the recorded on-device `benchmark_summary.csv`, using ONNX Runtime's **CPUExecutionProvider** (two inference threads, five trials, 500 timed inferences per trial). They measure **model-only inference**, not camera capture and display.

| Model | Median latency | Throughput | Peak process RAM |
|---|---:|---:|---:|
| FP32 baseline | 6.286 ms | 141.75 inf/s | 79.13 MB |
| Static INT8 | 4.133 ms | 248.30 inf/s | 69.75 MB |
| Unstructured 5% + INT8 | **3.448 ms** | **262.58 inf/s** | **69.14 MB** |
| Selective structured (6 groups) | 9.549 ms | 98.22 inf/s | 92.71 MB |
| Selective structured + INT8 | 5.100 ms | 182.83 inf/s | 74.04 MB |

**Interpretation:** 5% unstructured pruning + fine-tuning + static INT8 was the strongest observed **speed-and-storage** candidate: about **70.6% smaller** and **45.2% lower median inference latency** than the original FP32 baseline, with **87.60% test accuracy**. Its accuracy loss was **3.17 percentage points**, so it did **not** meet the initial 2-point threshold.

The selective structured FP32 model achieved the highest recorded test accuracy (93.40%) but was slower on this CPU backend. Its increased accuracy cannot be attributed to pruning alone: a fine-tuning-only control also improved validation accuracy. CPU performance differences do not prove the same ranking on GPU or other edge hardware.

## Results and reproducibility

The `results/` directory can hold a **curated** copy of experimental outputs, such as `results_extended.csv`, `sensitivity_results.csv`, `selected_groups.csv`, `selection_summary.json`, `final_model_test_comparison.csv`, and `benchmark_summary.csv`. These are not automatically read from `results/` by the scripts; experiments generate their own outputs under `outputs/`.

For each benchmark, record the device, OS, ONNX Runtime version, execution provider, thread count, number of warmups and repeats. Peak process RAM includes Python and runtime overhead. Report median and P95 latency and avoid treating theoretical MAC reduction as guaranteed wall-clock speedup.

## Known limitations and next steps

- TrashNet has a small, imbalanced miscellaneous-`trash` class and a limited image domain; evaluate per-class recall and live-camera generalization.
- The app classifies a **single dominant item**, not multiple objects or bounding boxes.
- INT8 did not meet the initial 2-percentage-point accuracy-loss target.
- The available benchmark is CPU-only; DirectML/GPU compatibility and actual operator placement require separate verification.
- End-to-end live-camera FPS and accuracy on a separate real-world camera test set have not yet been established in the shared results.
- Model and benchmark outputs should be checked for local absolute paths or identifying machine names before public upload.

## Dataset attribution

TrashNet dataset: https://github.com/garythung/trashnet . The pretrained MobileNetV2 backbone is supplied by Torchvision. Consult their respective licenses and attribution requirements before redistributing datasets or weights.
