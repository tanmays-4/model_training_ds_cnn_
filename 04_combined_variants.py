"""
Stage 4 -- Fix the NAS search + build the final missing combination
(NAS + Pruning + INT8), and produce a consolidated results table covering
every technique and combination for the report.

Run this AFTER 01_data_and_baseline.py, 02_compression_variants.py, and
03_convert_and_export.py (needs features.npz, baseline_model.h5,
pruned_model.h5, and the original arch_optimized_model.h5 / nas_search_results.csv
to exist, though it supersedes the latter two).

WHY THIS SCRIPT EXISTS
-----------------------
Your original nas_search_results.csv picked base_filters=32, num_blocks=4 as
the "best" architecture -- which is IDENTICAL to the baseline's own config.
That happened because:
  1. The search space only went down to 16 filters (not small enough to find
     a meaningfully different architecture), and
  2. `efficiency_score = val_acc - 0.05*(num_params/50000)` barely penalizes
     size difference at this scale (an 80% param reduction only costs ~0.06
     in score, easily outweighed by a 1-2% accuracy edge), so it degenerates
     into "just pick the most accurate candidate" -- i.e. not real NAS.

This script:
  (a) expands the search space to genuinely small architectures,
  (b) selects the winner under an explicit PARAMETER BUDGET constraint
      (best val_accuracy among candidates with params <= BUDGET_FRACTION
      of baseline's params) instead of a soft score, so the result is
      defensibly "NAS found a smaller model that trades some accuracy
      for size" -- which is the actual point of including NAS in your
      three-technique comparison,
  (c) applies pruning to that NAS-found architecture (NAS -> Pruned),
  (d) INT8-quantizes both the NAS-only and NAS+Pruned variants,
  (e) exports C headers for both, matching stage 3's convention,
  (f) writes a single consolidated results CSV covering every deployable
      configuration, ready to drop into your report.

Outputs:
  arch_optimized_model_v2.h5        (NAS winner, budget-constrained)
  nas_pruned_model.h5               (NAS winner + pruning + fine-tune)
  nas_search_results_expanded.csv   (every candidate tried, for your report)
  esp32_firmware/model_nas_v2.h
  esp32_firmware/model_nas_pruned.h
  combined_results_summary.csv      (consolidated table: all variants)
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # must come before `import tensorflow`

import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models

OUT_DIR = "esp32_firmware"
os.makedirs(OUT_DIR, exist_ok=True)

BUDGET_FRACTION = 0.5     # NAS winner must have <= 50% of baseline's params
SEARCH_EPOCHS = 8
FINAL_EPOCHS = 40
PRUNE_TARGET_SPARSITY = 0.5

# ---------------------------------------------------------------------------
# Load data + existing models
# ---------------------------------------------------------------------------
data = np.load("features.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

INPUT_SHAPE = X_train.shape[1:]
NUM_CLASSES = int(y_train.max()) + 1

baseline_model = tf.keras.models.load_model("baseline_model.h5")
baseline_acc = baseline_model.evaluate(X_test, y_test, verbose=0)[1]
baseline_params = baseline_model.count_params()
print(f"Baseline: acc={baseline_acc:.4f}  params={baseline_params}")

PARAM_BUDGET = int(baseline_params * BUDGET_FRACTION)
print(f"NAS parameter budget: <= {PARAM_BUDGET} params "
      f"({BUDGET_FRACTION:.0%} of baseline)")


# ===========================================================================
# 1. EXPANDED NAS SEARCH -- genuinely small candidates this time
# ===========================================================================
print("\n--- Expanded lightweight NAS search ---")

SEARCH_SPACE = [
    {"base_filters": 8,  "num_blocks": 2},
    {"base_filters": 8,  "num_blocks": 3},
    {"base_filters": 12, "num_blocks": 2},
    {"base_filters": 12, "num_blocks": 3},
    {"base_filters": 16, "num_blocks": 2},
    {"base_filters": 16, "num_blocks": 3},
    {"base_filters": 16, "num_blocks": 4},
    {"base_filters": 20, "num_blocks": 3},
    {"base_filters": 24, "num_blocks": 3},
]


def ds_conv_block(x, filters, strides=(1, 1)):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                                padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_ds_cnn(input_shape, num_classes, base_filters, num_blocks):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(base_filters, kernel_size=3, strides=(2, 2),
                       padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for i in range(num_blocks):
        strides = (2, 2) if i == 0 else (1, 1)
        x = ds_conv_block(x, filters=base_filters, strides=strides)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


results = []
for config in SEARCH_SPACE:
    print(f"\nTrying config: {config}")
    candidate = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES, **config)
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    candidate.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=SEARCH_EPOCHS,
        batch_size=64,
        verbose=0,
    )
    val_acc = candidate.evaluate(X_val, y_val, verbose=0)[1]
    num_params = candidate.count_params()
    within_budget = num_params <= PARAM_BUDGET

    results.append({
        "base_filters": config["base_filters"],
        "num_blocks": config["num_blocks"],
        "val_accuracy": val_acc,
        "num_params": num_params,
        "within_budget": within_budget,
    })
    print(f"  val_acc={val_acc:.4f}  params={num_params}  "
          f"within_budget={within_budget}")

with open("nas_search_results_expanded.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print("\nSaved nas_search_results_expanded.csv (full Pareto table for your report)")

# Select winner: best val_accuracy among candidates within the param budget.
in_budget = [r for r in results if r["within_budget"]]
if not in_budget:
    print("WARNING: no candidate met the param budget -- falling back to the "
          "smallest candidate searched. Consider widening SEARCH_SPACE further.")
    winner = min(results, key=lambda r: r["num_params"])
else:
    winner = max(in_budget, key=lambda r: r["val_accuracy"])

best_config = {"base_filters": winner["base_filters"], "num_blocks": winner["num_blocks"]}
print(f"\nSelected NAS winner (budget-constrained): {best_config}  "
      f"(val_acc={winner['val_accuracy']:.4f}, params={winner['num_params']}, "
      f"{100*winner['num_params']/baseline_params:.1f}% of baseline size)")

print("\nRetraining winner with full training budget...")
nas_model = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES, **best_config)
nas_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
nas_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINAL_EPOCHS,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ],
)
nas_test_acc = nas_model.evaluate(X_test, y_test, verbose=0)[1]
nas_params = nas_model.count_params()
print(f"NAS-v2 test accuracy: {nas_test_acc:.4f}  "
      f"(delta vs baseline: {nas_test_acc - baseline_acc:+.4f})  "
      f"params={nas_params}")

nas_model.save("arch_optimized_model_v2.h5")
print("Saved arch_optimized_model_v2.h5")


# ===========================================================================
# 2. NAS + PRUNING -- apply magnitude pruning to the NAS-found architecture
# ===========================================================================
print("\n--- Pruning the NAS-found architecture ---")

prune_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=PRUNE_TARGET_SPARSITY,
    begin_step=0,
    end_step=int(len(X_train) / 64) * 10,
)

prunable_nas = tfmot.sparsity.keras.prune_low_magnitude(
    nas_model,
    pruning_schedule=prune_schedule,
)
prunable_nas.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
prunable_nas.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
)

nas_pruned_model = tfmot.sparsity.keras.strip_pruning(prunable_nas)
nas_pruned_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
nas_pruned_acc = nas_pruned_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"NAS+Pruned ({PRUNE_TARGET_SPARSITY:.0%} sparsity) test accuracy: "
      f"{nas_pruned_acc:.4f}  (delta vs baseline: {nas_pruned_acc - baseline_acc:+.4f})")

nas_pruned_model.save("nas_pruned_model.h5")
print("Saved nas_pruned_model.h5")


# ===========================================================================
# 3. INT8-quantize both new variants + export C headers
# ===========================================================================
print("\n--- Quantizing + exporting NAS-v2 and NAS+Pruned ---")


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
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero_point = input_details["quantization"]
    correct = 0
    for i in range(len(X)):
        x_int8 = (X[i:i + 1] / in_scale + in_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], x_int8)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if np.argmax(out[0]) == y[i]:
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
    print(f"Wrote {out_path}  ({len(tflite_bytes)} bytes -> C array)")


nas_v2_int8 = quantize_keras_model(nas_model)
nas_v2_int8_acc = evaluate_tflite_int8(nas_v2_int8, X_test, y_test)
write_c_header(nas_v2_int8, "g_nas_v2_model", os.path.join(OUT_DIR, "model_nas_v2.h"))

nas_pruned_int8 = quantize_keras_model(nas_pruned_model)
nas_pruned_int8_acc = evaluate_tflite_int8(nas_pruned_int8, X_test, y_test)
write_c_header(nas_pruned_int8, "g_nas_pruned_model",
               os.path.join(OUT_DIR, "model_nas_pruned.h"))

print(f"\nNAS-v2 INT8 accuracy:        {nas_v2_int8_acc:.4f}")
print(f"NAS-v2 + Pruned INT8 accuracy: {nas_pruned_int8_acc:.4f}")


# ===========================================================================
# 4. Consolidated results table -- this is the table for your report
# ===========================================================================
# Pull in numbers from earlier stages where available (hardcode-free: reload
# the already-produced artifacts so this script is the single source of
# truth even if stage 2/3 were run in a different session).
pruned_model = tf.keras.models.load_model("pruned_model.h5")
pruned_acc = pruned_model.evaluate(X_test, y_test, verbose=0)[1]
pruned_params = pruned_model.count_params()

with open("quantized_model.tflite", "rb") as f:
    baseline_int8_bytes = f.read()
baseline_int8_acc = evaluate_tflite_int8(baseline_int8_bytes, X_test, y_test)

pruned_int8 = quantize_keras_model(pruned_model)
pruned_int8_acc = evaluate_tflite_int8(pruned_int8, X_test, y_test)

rows = [
    {"config": "baseline_fp32", "quantized": False, "pruned": False, "nas": False,
     "accuracy": baseline_acc, "params": baseline_params,
     "tflite_bytes": None, "deployed_on_device": "reference only"},
    {"config": "baseline_int8", "quantized": True, "pruned": False, "nas": False,
     "accuracy": baseline_int8_acc, "params": baseline_params,
     "tflite_bytes": len(baseline_int8_bytes), "deployed_on_device": "yes"},
    {"config": "pruned_int8", "quantized": True, "pruned": True, "nas": False,
     "accuracy": pruned_int8_acc, "params": pruned_params,
     "tflite_bytes": len(pruned_int8), "deployed_on_device": "yes"},
    {"config": "nas_v2_int8", "quantized": True, "pruned": False, "nas": True,
     "accuracy": nas_v2_int8_acc, "params": nas_params,
     "tflite_bytes": len(nas_v2_int8), "deployed_on_device": "yes"},
    {"config": "nas_v2_pruned_int8", "quantized": True, "pruned": True, "nas": True,
     "accuracy": nas_pruned_int8_acc, "params": nas_params,
     "tflite_bytes": len(nas_pruned_int8), "deployed_on_device": "yes"},
]

with open("combined_results_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved combined_results_summary.csv -- accuracy/size for every "
      "deployable configuration. Latency/RAM/energy come from the ESP32 "
      "firmware pass (see main.ino) -- merge those numbers into this table "
      "with 05_aggregate_results.py.")
print("\nAll headers ready in ./esp32_firmware/: "
      "model_baseline.h, model_baseline_float32.h, model_quantized.h, "
      "model_pruned.h, model_nas_v2.h, model_nas_pruned.h")
