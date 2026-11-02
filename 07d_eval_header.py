"""
Stage 7d -- Evaluate the model that is ACTUALLY inside a C header file, in
Python, to find out whether the ESP32 is running the same model we trained.

For each of model_nas_v2.h / model_nas_pruned.h (in ./esp32_firmware/):
  1. extract the byte array from the header text,
  2. check its length against the header's own *_len constant,
  3. compare it byte-for-byte with a fresh INT8 quantization of the matching
     .h5 file (prints whether identical + CRC32 of both),
  4. run the header's model in Python (reference kernels) and print
     "# INQ" / "# OUT" lines for the same 10 samples as the firmware dump,
     plus accuracy on the 50 samples and on the full test set.

Run (stdout only):
    !python 07d_eval_header.py 2>/dev/null
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import re
import zlib
import numpy as np
import tensorflow as tf

HEADERS = {
    "nas_v2_int8": ("esp32_firmware/model_nas_v2.h", "arch_optimized_model_v2.h5"),
    "nas_v2_pruned_int8": ("esp32_firmware/model_nas_pruned.h", "nas_pruned_model.h5"),
}

d = np.load("features.npz")
X_train, X_test, y_test = d["X_train"], d["X_test"], d["y_test"]
idx = []
for c in range(10):
    idx += np.where(y_test == c)[0][:5].tolist()
X50, y50 = X_test[idx], y_test[idx]


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


def parse_header(path):
    text = open(path).read()
    body = text[text.index("{"):text.index("};")]
    data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]{2})", body))
    declared = int(re.search(r"_len\s*=\s*(\d+)", text).group(1))
    return data, declared


def run(tflite_bytes, X):
    it = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    outs = np.zeros((len(X), out["shape"][-1]), dtype=np.int64)
    for i in range(len(X)):
        q = np.clip(np.round(X[i:i + 1] / scale) + zp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
        it.invoke()
        outs[i] = it.get_tensor(out["index"])[0]
    return outs, scale, int(zp)


for name, (hpath, h5path) in HEADERS.items():
    print(f"\n=== {name}  ({hpath}) ===")
    hbytes, declared = parse_header(hpath)
    print(f"# bytes parsed={len(hbytes)}  declared _len={declared}  "
          f"match={len(hbytes) == declared}  crc32={zlib.crc32(hbytes):08x}")
    fresh = quantize_keras_model(h5path)
    print(f"# fresh quantization of {h5path}: {len(fresh)} bytes  "
          f"crc32={zlib.crc32(fresh):08x}  identical_to_header={fresh == hbytes}")

    outs50, scale, zp = run(hbytes, X50)
    print(f"# INQ,{name},{scale:.8f},{zp}")
    for i in range(0, 50, 5):
        print(f"# OUT,{name},{i}," + ",".join(str(int(v)) for v in outs50[i]))
    acc50 = float((outs50.argmax(1) == y50).mean())
    outs_full, _, _ = run(hbytes, X_test)
    acc_full = float((outs_full.argmax(1) == y_test).mean())
    print(f"# header-model accuracy: 50-sample={acc50:.2f}  full-test={acc_full:.4f}")
