"""
Stage 7c -- Print the raw INT8 output scores and input quantization params that
the firmware prints in DEBUG_DUMP mode, computed in Python with reference
kernels, for the same samples (sample i = first 5 per class in X_test, one per
class: i = 0, 5, 10, ..., 45). Output lines use the same format as the
firmware's "# INQ" / "# OUT" lines so the two can be diffed.

Run (stdout only):
    !python 07c_dump_int8_outputs.py 2>/dev/null
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

d = np.load("features.npz")
X_train, X_test, y_test = d["X_train"], d["X_test"], d["y_test"]
idx = []
for c in range(10):
    idx += np.where(y_test == c)[0][:5].tolist()
X50 = X_test[idx]


def representative_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i + 1].astype(np.float32)]


def quantize_keras_model(path):
    m = tf.keras.models.load_model(path)
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


with open("quantized_model.tflite", "rb") as f:
    baseline_bytes = f.read()

models = {
    "baseline_int8": baseline_bytes,
    "pruned_int8": quantize_keras_model("pruned_model.h5"),
    "nas_v2_int8": quantize_keras_model("arch_optimized_model_v2.h5"),
    "nas_v2_pruned_int8": quantize_keras_model("nas_pruned_model.h5"),
}

for name, b in models.items():
    it = tf.lite.Interpreter(
        model_content=b,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    print(f"# INQ,{name},{scale:.8f},{int(zp)}")
    for i in range(0, 50, 5):
        q = np.clip(np.round(X50[i:i + 1] / scale) + zp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
        it.invoke()
        o = it.get_tensor(out["index"])[0]
        print(f"# OUT,{name},{i}," + ",".join(str(int(v)) for v in o))
