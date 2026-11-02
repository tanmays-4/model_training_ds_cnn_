"""
Stage 6 -- Evaluate all deployed model variants on every test set you have:
the main clean test set, the clean/noisy sets from 00_build_noisy_testset.py,
and any extra .npz test sets you pass on the command line (e.g. a Speech
Commands v0.01 test set built the same way; each needs X_test and y_test).

NO TRAINING. Models are loaded from the .h5 / .tflite files already produced,
INT8-quantized exactly as in stages 3/4 (same calibration data), and scored.

Run (from the folder holding the models + npz files):
    !python 06_eval_robustness.py
    !python 06_eval_robustness.py extra_v001_test.npz     # optional extras

Writes robustness_results.csv (model x dataset accuracy) and prints a table.
Compare noisy sets against clean_test_features.npz (same file list), not
against features.npz.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import sys
import numpy as np
import tensorflow as tf

X_train = np.load("features.npz")["X_train"]


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


def load_h5(path):
    m = tf.keras.models.load_model(path)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def eval_int8(tflite_bytes, X, y):
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    correct = 0
    for i in range(len(X)):
        it.set_tensor(inp["index"], (X[i:i + 1] / scale + zp).astype(np.int8))
        it.invoke()
        correct += int(np.argmax(it.get_tensor(out["index"])[0]) == y[i])
    return correct / len(X)


def eval_fp32(model, X, y):
    return float((np.argmax(model.predict(X, verbose=0), axis=1) == y).mean())


# name -> callable(X, y) -> accuracy
print("Preparing models...")
baseline = load_h5("baseline_model.h5")
with open("quantized_model.tflite", "rb") as f:
    baseline_int8 = f.read()
pruned_int8 = quantize_keras_model(load_h5("pruned_model.h5"))
nas_int8 = quantize_keras_model(load_h5("arch_optimized_model_v2.h5"))
nas_pruned_int8 = quantize_keras_model(load_h5("nas_pruned_model.h5"))

MODELS = {
    "baseline_fp32": lambda X, y: eval_fp32(baseline, X, y),
    "baseline_int8": lambda X, y: eval_int8(baseline_int8, X, y),
    "pruned_int8": lambda X, y: eval_int8(pruned_int8, X, y),
    "nas_v2_int8": lambda X, y: eval_int8(nas_int8, X, y),
    "nas_v2_pruned_int8": lambda X, y: eval_int8(nas_pruned_int8, X, y),
}

candidates = ["features.npz", "clean_test_features.npz",
              "noisy_test_features_10db.npz", "noisy_test_features_0db.npz"]
candidates += sys.argv[1:]
datasets = [p for p in candidates if os.path.exists(p)]
missing = [p for p in candidates if not os.path.exists(p)]
if missing:
    print(f"Skipping (not found): {missing}")

rows = []
for path in datasets:
    d = np.load(path)
    X, y = d["X_test"], d["y_test"]
    print(f"\n{path}: {len(X)} samples")
    for name, fn in MODELS.items():
        acc = fn(X, y)
        rows.append({"model": name, "dataset": path, "n": len(X),
                     "accuracy": round(acc, 4)})
        print(f"  {name:22s} {acc:.4f}")

with open("robustness_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "dataset", "n", "accuracy"])
    w.writeheader()
    w.writerows(rows)
print("\nWrote robustness_results.csv")
