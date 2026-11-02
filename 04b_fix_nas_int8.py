"""
Stage 4b -- Fix for the in-place pruning bug in 04_combined_variants.py.
NO TRAINING. Takes ~1 minute.

Re-quantizes the NAS-only and NAS+Pruned models from the .h5 files that were
saved BEFORE the bug could affect them, re-exports their C headers, prints the
real weight sparsity of each model (sanity check), and updates
combined_results_summary.csv -- including the fp32 rows for the pairwise
combinations (pruning only, NAS only, NAS+pruning without quantization).

Run from your Drive project folder:
    !python 04b_fix_nas_int8.py
Needs: features.npz, arch_optimized_model_v2.h5, nas_pruned_model.h5,
       pruned_model.h5, combined_results_summary.csv (from 04).
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # before importing tensorflow

import csv
import numpy as np
import tensorflow as tf

OUT_DIR = "esp32_firmware"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = "combined_results_summary.csv"

data = np.load("features.npz")
X_train = data["X_train"]
X_test, y_test = data["X_test"], data["y_test"]


def load(path):
    m = tf.keras.models.load_model(path)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def kernel_sparsity(model):
    """Fraction of exactly-zero weights in conv/dense kernels (ndim >= 2)."""
    total = zeros = 0
    for w in model.get_weights():
        if w.ndim >= 2:
            total += w.size
            zeros += int((w == 0).sum())
    return zeros / total


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


def evaluate_tflite_int8(tflite_model, X, y):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zp = inp["quantization"]
    correct = 0
    for i in range(len(X)):
        x_int8 = (X[i:i + 1] / scale + zp).astype(np.int8)
        interpreter.set_tensor(inp["index"], x_int8)
        interpreter.invoke()
        if np.argmax(interpreter.get_tensor(out["index"])[0]) == y[i]:
            correct += 1
    return correct / len(X)


def write_c_header(tflite_bytes, var_name, out_path):
    with open(out_path, "w") as f:
        f.write(f"// Auto-generated from {var_name}. Do not edit by hand.\n")
        f.write("#pragma once\n\n#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print(f"Wrote {out_path} ({len(tflite_bytes)} bytes)")


nas = load("arch_optimized_model_v2.h5")
nas_pruned = load("nas_pruned_model.h5")
pruned = load("pruned_model.h5")

sp_nas, sp_nas_pruned, sp_pruned = (
    kernel_sparsity(nas), kernel_sparsity(nas_pruned), kernel_sparsity(pruned))
print("\nKernel sparsity (zero-weight fraction):")
print(f"  NAS-v2 (should be ~0):        {sp_nas:.3f}")
print(f"  NAS-v2 + pruned (~0.5):       {sp_nas_pruned:.3f}")
print(f"  baseline pruned (~0.5):       {sp_pruned:.3f}")
if sp_nas > 0.05:
    print("  WARNING: arch_optimized_model_v2.h5 is already sparse -- it was "
          "saved after pruning. Rerun the patched 04_combined_variants.py.")

acc_nas_fp32 = nas.evaluate(X_test, y_test, verbose=0)[1]
acc_nas_pruned_fp32 = nas_pruned.evaluate(X_test, y_test, verbose=0)[1]
acc_pruned_fp32 = pruned.evaluate(X_test, y_test, verbose=0)[1]

nas_int8 = quantize_keras_model(nas)
nas_pruned_int8 = quantize_keras_model(nas_pruned)
acc_nas_int8 = evaluate_tflite_int8(nas_int8, X_test, y_test)
acc_nas_pruned_int8 = evaluate_tflite_int8(nas_pruned_int8, X_test, y_test)

write_c_header(nas_int8, "g_nas_v2_model", os.path.join(OUT_DIR, "model_nas_v2.h"))
write_c_header(nas_pruned_int8, "g_nas_pruned_model",
               os.path.join(OUT_DIR, "model_nas_pruned.h"))

print("\nCorrected accuracies:")
print(f"  NAS-v2 fp32:            {acc_nas_fp32:.4f}")
print(f"  NAS-v2 INT8:            {acc_nas_int8:.4f}")
print(f"  NAS-v2+pruned fp32:     {acc_nas_pruned_fp32:.4f}")
print(f"  NAS-v2+pruned INT8:     {acc_nas_pruned_int8:.4f}")
if abs(acc_nas_int8 - acc_nas_pruned_int8) < 1e-9:
    print("  WARNING: NAS INT8 and NAS+pruned INT8 are still identical -- "
          "paste this output to Claude.")

# ---- update the summary table ----------------------------------------------
FIELDS = ["config", "quantized", "pruned", "nas", "accuracy", "params",
          "tflite_bytes", "kernel_sparsity", "deployed_on_device"]
with open(SUMMARY, newline="") as f:
    rows = {r["config"]: r for r in csv.DictReader(f)}
order = list(rows.keys())

nas_params, pruned_params = nas.count_params(), pruned.count_params()
updates = [
    dict(config="pruned_fp32", quantized=False, pruned=True, nas=False,
         accuracy=acc_pruned_fp32, params=pruned_params, tflite_bytes="",
         kernel_sparsity=sp_pruned, deployed_on_device="no (accuracy only)"),
    dict(config="nas_v2_fp32", quantized=False, pruned=False, nas=True,
         accuracy=acc_nas_fp32, params=nas_params, tflite_bytes="",
         kernel_sparsity=sp_nas, deployed_on_device="no (accuracy only)"),
    dict(config="nas_v2_pruned_fp32", quantized=False, pruned=True, nas=True,
         accuracy=acc_nas_pruned_fp32, params=nas_params, tflite_bytes="",
         kernel_sparsity=sp_nas_pruned, deployed_on_device="no (accuracy only)"),
    dict(config="nas_v2_int8", quantized=True, pruned=False, nas=True,
         accuracy=acc_nas_int8, params=nas_params, tflite_bytes=len(nas_int8),
         kernel_sparsity=sp_nas, deployed_on_device="yes"),
    dict(config="nas_v2_pruned_int8", quantized=True, pruned=True, nas=True,
         accuracy=acc_nas_pruned_int8, params=nas_params,
         tflite_bytes=len(nas_pruned_int8), kernel_sparsity=sp_nas_pruned,
         deployed_on_device="yes"),
]
for u in updates:
    base = rows.get(u["config"], {})
    base.update(u)
    rows[u["config"]] = base
    if u["config"] not in order:
        order.append(u["config"])

with open(SUMMARY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
    w.writeheader()
    for name in order:
        w.writerow(rows[name])
print(f"\nUpdated {SUMMARY}")
