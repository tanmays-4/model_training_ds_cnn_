# Model Compression for Keyword Spotting on ESP32-WROOM-32

Code, trained models, firmware and results for the paper

> **A Systematic Empirical Study of Model Compression and Architecture Optimization Techniques for Neural Network Deployment on Resource-Constrained Microcontrollers**
> Kannoth, Sharma — Vellore Institute of Technology, Chennai

We benchmark three compression techniques, alone and in combination, on a depthwise-separable CNN (DS-CNN) for 10-class keyword spotting, and measure accuracy, latency and memory on a real ESP32-WROOM-32 running TensorFlow Lite Micro.

| Technique | Implementation |
|---|---|
| INT8 quantization | Post-training, TFLite converter, 200-sample representative set |
| Pruning | Unstructured magnitude pruning, TF-MOT `prune_low_magnitude` (measured ~37–40 % zero weights) |
| NAS | Budget-constrained grid search over filters (8–24) × blocks (2–4), budget = 50 % of baseline parameters |

Evaluated variants: baseline, pruned, NAS, NAS + pruned — each in FP32 and INT8.
the features.npz can be found in https://drive.google.com/drive/folders/1BMxPLaFmkS1fgYz3ZnBhnX4DNtkNPfxi?usp=sharing
---

## 1. Dataset

The raw audio is public and is not stored in this repository.

- **Google Speech Commands v0.02** (Warden, 2018, [arXiv:1804.03209](https://arxiv.org/abs/1804.03209)), CC BY 4.0
  Download: <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz> (~2.3 GB)
- **Split:** the dataset's own `validation_list.txt` and `testing_list.txt` (speaker-disjoint); everything else is training.
- **Classes (10):** yes, no, up, down, stop, go, left, right, `unknown` (all other words, capped at 4,000 / 1,000 / 1,000 for train / val / test), `silence` (random 1 s crops of `_background_noise_`, 2,000 / 500 / 500).
- **Features:** 40 MFCCs × 49 frames, 16 kHz, n_fft 512, hop 320, per-sample zero-mean / unit-variance. Input tensor 49 × 40 × 1.
- **Sizes:** train 30,713 / validation 4,467 / test 4,776. Seed 42.
- **`features.npz`** (derived MFCC arrays, ~264 MB) is too large for git. Regenerate it with `01_data_and_baseline.py`, or download it here: **<ADD LINK>**.
- **On-device test set:** the first 5 test examples of each class (50 in total), exported as float32 to `esp32_firmware/esp32_firmware_main/test_samples.h`.

## 2. Repository contents

| File | Purpose |
|---|---|
| `01_data_and_baseline.py` | Downloads the dataset, extracts MFCCs (`features.npz`), trains the baseline DS-CNN |
| `02_compression_variants.py` | INT8 PTQ, pruning, first NAS search |
| `03_convert_and_export.py`, `03_convert_and_export_modified.py` | TFLite conversion and C-header export for the board |
| `04_combined_variants.py` | Budget-constrained NAS (v2), NAS + pruning, INT8 export, `combined_results_summary.csv` |
| `04b_fix_nas_int8.py` | Re-quantizes the NAS models from the saved `.h5` files (kept for provenance, see note below) |
| `00_build_noisy_testset.py`, `06_eval_robustness.py` | Noisy-audio test sets (clean / 10 dB / 0 dB) and evaluation |
| `05_aggregate_results.py` | Merges offline and on-device results into the report table |
| `07_*.py`, `07b_*.py`, `07c_*.py`, `07d_*.py` | Python-vs-ESP32 consistency diagnostics |
| `08_make_bisect_models.py` | Layer-by-layer test models for locating where the ESP32 output diverges |
| `esp32_firmware/` | Arduino sketch, generated model headers, test samples, bisection sketch |
| `*.h5`, `*.tflite` | Trained Keras and INT8 models |
| `*.csv` | NAS search results, summary table, ESP32 serial log |
| `model_embedded_project.ipynb` | Colab notebook used to run the pipeline |

> Note: an early version of `04_combined_variants.py` pruned the NAS model in place, which made the NAS and NAS + pruned INT8 results identical. The current `04` prunes a copy. `04b` re-quantized the affected models from the saved `.h5` files, and the reported numbers come from the corrected models.

## 3. How to reproduce

**Software:** Google Colab (T4 GPU) for training and conversion.

```bash
pip install tensorflow_model_optimization tf_keras librosa   # then restart the runtime
export TF_USE_LEGACY_KERAS=1                                 # must be set before importing TensorFlow
```

Run in order: `01` → `02` → `03_convert_and_export_modified` → `04` → `04b` → `05`.
Record exact versions with `pip freeze | grep -E "tensorflow|tf_keras|librosa|numpy"` (see `environment.txt`).

**Firmware** (`esp32_firmware/esp32_firmware_main/`):

- Arduino IDE 1.8.19, board **ESP32 Dev Module**, 240 MHz, partition **Huge APP (3 MB No OTA / 1 MB SPIFFS)**
- Library: **Chirale_TensorFlowLite 2.0.0**; ESP32 board package: **<ADD VERSION>**
- Set `DEBUG_DUMP 0`, open the Serial Monitor at 115200, press EN **once** and wait for `DONE` (about 40 s). The CSV line is printed per model.
- The tensor arena is allocated from the heap in `setup()` (largest free block ≈ 110 KB).
- Latency is measured with `micros()` around `Invoke()` only; MFCC features are precomputed, not computed on the device.

## 4. Results

### Accuracy on the full test set (4,776 examples, offline Python)

| Model | FP32 | INT8 |
|---|---|---|
| Baseline | 90.89 % | 89.87 % |
| Pruned | 89.01 % | 88.48 % |
| NAS v2 | 85.26 % | 84.48 % |
| NAS v2 + pruned | 80.40 % | 79.61 % |

### On-device (ESP32), INT8 models, 50 test samples

| Model | Python, same 50 samples | ESP32 |
|---|---|---|
| Baseline | 94 % | 94 % |
| Pruned | 84 % | 88 % |
| NAS v2 | 90 % | 64 % |
| NAS v2 + pruned | 78 % | 72 % |

### Latency and memory (stable from run to run)

- INT8 is about 5.5× faster than FP32 and uses about 70 % less tensor arena.
- NAS v2 is about 2.1× faster than the INT8 baseline, with about 37 % less arena and 39 % less flash.
- Unstructured pruning gives **no latency, flash or arena benefit** on the device, because the model is stored dense.

Full numbers: `combined_results_summary.csv` and `esp32_serial_log.csv`.

## 5. Known limitations

- **On-device accuracy uses only 50 samples per model**, so it is preliminary and each sample is worth 2 percentage points.
- **Unresolved Python-vs-ESP32 mismatch.** The model bytes on the device match the headers (hash-verified), the quantized inputs match, and upstream TFLM reproduces TFLite reference kernels bit-for-bit on a PC. Yet the ESP32 output differs for several models (largest for NAS v2 INT8: 92 % in Python on the same samples, 64 % on the device). We are isolating the first diverging layer with `08_make_bisect_models.py` and `esp32_firmware/esp32_bisect/`. Until this is resolved, the on-device accuracy figures should be read as measurements of this specific ESP32 build, not of the models themselves. The latency and memory results are not affected.
- Pruning is unstructured, with about 40 % measured sparsity rather than the 50 % target.
- No energy measurements yet.
- Noisy-audio robustness scripts are included but results are not yet validated.

## 6. Licence and citation

Code: **<ADD LICENCE, e.g. MIT>**.
Dataset: Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 (CC BY 4.0).

If you use this work, please cite the paper above. **<ADD BIBTEX AFTER PUBLICATION>**
