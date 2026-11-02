"""
Stage 7 -- Sanity check: does the ESP32 give the same answers as Python?

The firmware scored 5 test samples per class (50 total, the first 5 of each
class in X_test, exactly as 03_convert_and_export_modified.py exported them).
This script scores the SAME 50 samples in Python, with input quantization done
the way the firmware does it (round-to-nearest), and also reports:
  - overall accuracy on the full test set (class mix is imbalanced), and
  - balanced accuracy = mean per-class recall on the full test set, which is
    the fair thing to compare against a 5-per-class sample.

Outcomes:
  * Python-50 == device accuracy for every model  -> firmware is correct; any
    gap vs the headline accuracy is sampling / class-mix, not a bug.
  * Python-50 differs from device                  -> deployment bug; paste the
    output back.

Run in Colab from the project folder (packages installed, runtime restarted):
    !python 07_check_50_samples.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

CLASSES = ["yes", "no", "up", "down", "stop", "go", "left", "right",
           "unknown", "silence"]
DEVICE = {  # from esp32_serial_log.csv
    "baseline_fp32": 0.94, "baseline_int8": 0.94, "pruned_int8": 0.88,
    "nas_v2_int8": 0.64, "nas_v2_pruned_int8": 0.72,
}

d = np.load("features.npz")
X_train, X_test, y_test = d["X_train"], d["X_test"], d["y_test"]

idx = []
for c in range(len(CLASSES)):
    idx += np.where(y_test == c)[0][:5].tolist()
X50, y50 = X_test[idx], y_test[idx]


def representative_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i + 1].astype(np.float32)]


def quantize_keras_model(m):
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def load_h5(p):
    return tf.keras.models.load_model(p)


def predict_int8(tflite_bytes, X):
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    preds = np.zeros(len(X), dtype=np.int64)
    for i in range(len(X)):
        q = np.clip(np.round(X[i:i + 1] / scale) + zp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
        it.invoke()
        preds[i] = np.argmax(it.get_tensor(out["index"])[0])
    return preds


baseline = load_h5("baseline_model.h5")
with open("quantized_model.tflite", "rb") as f:
    baseline_int8 = f.read()

predictors = {
    "baseline_fp32": lambda X: np.argmax(baseline.predict(X, verbose=0), axis=1),
    "baseline_int8": lambda X: predict_int8(baseline_int8, X),
    "pruned_int8": lambda X: predict_int8(
        quantize_keras_model(load_h5("pruned_model.h5")), X),
    "nas_v2_int8": lambda X: predict_int8(
        quantize_keras_model(load_h5("arch_optimized_model_v2.h5")), X),
    "nas_v2_pruned_int8": lambda X: predict_int8(
        quantize_keras_model(load_h5("nas_pruned_model.h5")), X),
}

print(f"{'model':22s} {'py-50':>6s} {'device':>7s} {'full-test':>10s} {'balanced':>9s}")
per_class_rows = {}
for name, fn in predictors.items():
    p50 = fn(X50)
    acc50 = float((p50 == y50).mean())
    pfull = fn(X_test)
    overall = float((pfull == y_test).mean())
    recalls = [float((pfull[y_test == c] == c).mean()) for c in range(len(CLASSES))]
    balanced = float(np.mean(recalls))
    per_class_rows[name] = recalls
    print(f"{name:22s} {acc50:6.2f} {DEVICE[name]:7.2f} {overall:10.4f} {balanced:9.4f}")

print("\nPer-class recall on the full test set:")
print(f"{'model':22s} " + " ".join(f"{c[:5]:>5s}" for c in CLASSES))
for name, r in per_class_rows.items():
    print(f"{name:22s} " + " ".join(f"{v:5.2f}" for v in r))
