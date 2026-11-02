"""
Stage 3 — Convert every model variant into a C header array for the ESP32.

Run this AFTER 01_data_and_baseline.py and 02_compression_variants.py.

For fair comparison, ALL FOUR variants are converted through the same
INT8 post-training quantization path before going to the ESP32 -- otherwise
you'd be comparing float32 models against an INT8 model, which confounds
your results. The "quantized_model.tflite" from stage 2 is INT8 already;
here we also INT8-quantize baseline, pruned, and arch_optimized so every
row in your final comparison table used the same numeric precision on-device,
and the *architecture/pruning* effect is isolated from the *quantization*
effect.

If you specifically want to also report a float32-baseline latency number
(useful to show "quantization alone got us this much"), keep a float32
.tflite of the baseline around too -- this script produces one as
baseline_model_float32.h for that comparison.

Outputs (in ./esp32_firmware/):
  model_baseline.h
  model_baseline_float32.h
  model_quantized.h
  model_pruned.h
  model_arch_optimized.h
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # match the setting used when saving
                                            # pruned_model.h5 in step 2, so this
                                            # script loads it correctly. Must
                                            # come before `import tensorflow`.

import numpy as np
import tensorflow as tf

OUT_DIR = "esp32_firmware"
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load("features.npz")
X_train = data["X_train"]


def representative_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i + 1].astype(np.float32)]


def quantize_keras_model(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def float32_tflite(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return converter.convert()


def write_c_header(tflite_bytes, var_name, out_path):
    """Equivalent of `xxd -i`, written in Python so it works with no
    external tools -- just numpy/tensorflow, which you already have."""
    with open(out_path, "w") as f:
        f.write(f"// Auto-generated from {var_name}. Do not edit by hand.\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"  {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print(f"Wrote {out_path}  ({len(tflite_bytes)} bytes -> C array)")


# ---------------------------------------------------------------------------
# Baseline (INT8 + float32 for the "quantization alone" comparison)
# ---------------------------------------------------------------------------
baseline_model = tf.keras.models.load_model("baseline_model.h5")

baseline_int8 = quantize_keras_model(baseline_model)
write_c_header(baseline_int8, "g_baseline_model",
               os.path.join(OUT_DIR, "model_baseline.h"))

baseline_fp32 = float32_tflite(baseline_model)
write_c_header(baseline_fp32, "g_baseline_model_fp32",
               os.path.join(OUT_DIR, "model_baseline_float32.h"))

# ---------------------------------------------------------------------------
# Quantized (already produced directly as INT8 in stage 2 -- just re-export
# with the matching header/variable name convention)
# ---------------------------------------------------------------------------
with open("quantized_model.tflite", "rb") as f:
    quantized_bytes = f.read()
write_c_header(quantized_bytes, "g_quantized_model",
               os.path.join(OUT_DIR, "model_quantized.h"))

# ---------------------------------------------------------------------------
# Pruned
# ---------------------------------------------------------------------------
pruned_model = tf.keras.models.load_model("pruned_model.h5")
pruned_int8 = quantize_keras_model(pruned_model)
write_c_header(pruned_int8, "g_pruned_model",
               os.path.join(OUT_DIR, "model_pruned.h"))

# ---------------------------------------------------------------------------
# Architecture-optimized
# ---------------------------------------------------------------------------
arch_model = tf.keras.models.load_model("arch_optimized_model.h5")
arch_int8 = quantize_keras_model(arch_model)
write_c_header(arch_int8, "g_arch_optimized_model",
               os.path.join(OUT_DIR, "model_arch_optimized.h"))

print("\nAll model headers written to ./esp32_firmware/")

# ---------------------------------------------------------------------------
# Export a handful of test samples as FLOAT32 C arrays. We keep them as
# float and quantize on-device using each model's own input scale/zero-point
# (read from the TFLM interpreter at runtime) -- this is important because
# baseline/quantized/pruned/arch_optimized each get their own INT8 scale
# from the converter, so a single pre-quantized sample would be wrong for
# three of the four models.
# ---------------------------------------------------------------------------
X_test = data["X_test"] if "X_test" in data else np.load("features.npz")["X_test"]
y_test = np.load("features.npz")["y_test"]

SAMPLES_PER_CLASS = 5
CLASSES = ["yes", "no", "up", "down", "stop", "go", "left", "right",
           "unknown", "silence"]
N_TEST_SAMPLES = SAMPLES_PER_CLASS * len(CLASSES)

# Deterministically pick the first SAMPLES_PER_CLASS held-out test-set
# indices for each class (in CLASSES order), so the exact same 50 samples
# are selected every time this script runs against the same features.npz.
# Only X_test/y_test are touched here -- no train/val data, nothing generated.
selected_indices = []
class_counts = {c: 0 for c in CLASSES}
for class_idx, class_name in enumerate(CLASSES):
    matches = np.where(y_test == class_idx)[0]
    chosen = matches[:SAMPLES_PER_CLASS]
    if len(chosen) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Only found {len(chosen)} test samples for class '{class_name}' "
            f"(need {SAMPLES_PER_CLASS}). Check your test split."
        )
    selected_indices.extend(chosen.tolist())
    class_counts[class_name] = len(chosen)

with open(os.path.join(OUT_DIR, "test_samples.h"), "w") as f:
    f.write("// Auto-generated float32 MFCC test samples + true labels\n")
    f.write("// 5 samples per class, deterministically selected from the\n")
    f.write("// held-out X_test/y_test split (first 5 matching test indices\n")
    f.write("// per class, in CLASSES order). Re-running this script against\n")
    f.write("// the same features.npz always picks the same 50 samples.\n")
    f.write("#pragma once\n\n")
    f.write(f"#define NUM_TEST_SAMPLES {N_TEST_SAMPLES}\n")
    f.write(f"#define FEATURE_FRAMES {X_test.shape[1]}\n")
    f.write(f"#define FEATURE_COEFFS {X_test.shape[2]}\n")
    f.write(f"#define FEATURE_SIZE (FEATURE_FRAMES * FEATURE_COEFFS)\n\n")
    f.write("const char* kCategoryLabels[] = {\n  ")
    f.write(", ".join(f'"{c}"' for c in CLASSES))
    f.write("\n};\n\n")

    f.write("const float g_test_samples[NUM_TEST_SAMPLES][FEATURE_SIZE] = {\n")
    for idx in selected_indices:
        flat = X_test[idx, :, :, 0].flatten()
        line = ", ".join(f"{v:.6f}f" for v in flat)
        f.write(f"  {{ {line} }},\n")
    f.write("};\n\n")

    f.write("const int g_test_labels[NUM_TEST_SAMPLES] = {\n  ")
    f.write(", ".join(str(int(y_test[idx])) for idx in selected_indices))
    f.write("\n};\n")

print(f"Wrote {os.path.join(OUT_DIR, 'test_samples.h')}")
print(f"\nTotal samples: {N_TEST_SAMPLES}")
for class_name in CLASSES:
    print(f"{class_name}: {class_counts[class_name]}")

print("\nCopy the whole esp32_firmware/ folder into your Arduino sketch folder.")
