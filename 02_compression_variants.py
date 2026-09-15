"""
Stage 2 — Compression variants: quantization, pruning, lightweight NAS
Run this AFTER 01_data_and_baseline.py (reuses features.npz + baseline_model.h5)

Produces:
  quantized_model.tflite      (INT8 post-training quantization of the baseline)
  pruned_model.h5             (magnitude-pruned + fine-tuned baseline, unstructured sparsity)
  arch_optimized_model.h5     (best model found by the lightweight architecture search)
  nas_search_results.csv      (every architecture tried + its accuracy/params, for your report)

Note on pruning scope: this uses UNSTRUCTURED magnitude pruning (tfmot's
prune_low_magnitude), which zeros out individual weights without physically
removing them from the tensor. This is standard for measuring "how much
accuracy do we lose at X% sparsity", but on its own often does NOT reduce
ESP32 inference latency or RAM, because a zeroed weight still occupies memory
and gets multiplied unless the inference engine has specific sparse-matrix
support (TFLite Micro generally does not). This is exactly the kind of
finding your research question expects you to report on -- don't be
surprised if pruning "wins" on accuracy retention but not on latency/RAM.
Structured pruning (removing whole channels) is effectively explored by the
NAS-lite search below, since it directly varies channel counts.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tensorflow_model_optimization needs
                                            # Keras 2, not the Keras-3 default --
                                            # this line MUST come before `import
                                            # tensorflow`. If you already imported
                                            # tensorflow earlier in this Colab
                                            # session, restart the runtime first.

import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Load features + baseline model saved by stage 1
# ---------------------------------------------------------------------------
data = np.load("features.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

INPUT_SHAPE = X_train.shape[1:]        # (N_FRAMES, N_MFCC, 1)
NUM_CLASSES = int(y_train.max()) + 1

baseline_model = tf.keras.models.load_model("baseline_model.h5")
baseline_acc = baseline_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Baseline test accuracy (reference): {baseline_acc:.4f}")


# ===========================================================================
# 1. QUANTIZATION — INT8 post-training quantization
# ===========================================================================
def representative_dataset():
    """TFLite needs a small sample of real inputs to calibrate INT8 ranges."""
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i + 1].astype(np.float32)]


def quantize_model(keras_model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {out_path} ({len(tflite_model)} bytes)")
    return tflite_model


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
        pred = np.argmax(out[0])
        if pred == y[i]:
            correct += 1
    return correct / len(X)


print("\n--- Quantization ---")
quantized_tflite = quantize_model(baseline_model, "quantized_model.tflite")
quant_acc = evaluate_tflite_int8(quantized_tflite, X_test, y_test)
print(f"Quantized (INT8) test accuracy: {quant_acc:.4f}  "
      f"(delta vs baseline: {quant_acc - baseline_acc:+.4f})")


# ===========================================================================
# 2. PRUNING — unstructured magnitude pruning + fine-tuning
# ===========================================================================
print("\n--- Pruning ---")

PRUNE_TARGET_SPARSITY = 0.5   # 50% of weights zeroed -- adjust and re-run to
                               # sweep sparsity levels for your report

prune_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=PRUNE_TARGET_SPARSITY,
    begin_step=0,
    end_step=int(len(X_train) / 64) * 10,   # ~10 epochs worth of steps
)

# IMPORTANT:
# The baseline model is already trained. TensorFlow Model Optimization's
# pruning API expects the pretrained weights to be loaded BEFORE applying
# prune_low_magnitude(). Wrapping a fresh clone and then calling set_weights()
# causes a weight-count mismatch because pruning wrappers add mask/threshold
# variables.
prunable_model = tfmot.sparsity.keras.prune_low_magnitude(
    baseline_model,
    pruning_schedule=prune_schedule,
)

prunable_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

prunable_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
)

pruned_model = tfmot.sparsity.keras.strip_pruning(prunable_model)
pruned_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
pruned_acc = pruned_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Pruned ({PRUNE_TARGET_SPARSITY:.0%} sparsity) test accuracy: {pruned_acc:.4f}  "
      f"(delta vs baseline: {pruned_acc - baseline_acc:+.4f})")

pruned_model.save("pruned_model.h5")
print("Saved pruned_model.h5")


# ===========================================================================
# 3. LIGHTWEIGHT NAS — grid search over architecture knobs
# ===========================================================================
print("\n--- Lightweight NAS search ---")

SEARCH_SPACE = [
    {"base_filters": 16, "num_blocks": 3},
    {"base_filters": 16, "num_blocks": 4},
    {"base_filters": 24, "num_blocks": 3},
    {"base_filters": 24, "num_blocks": 4},
    {"base_filters": 32, "num_blocks": 3},
    {"base_filters": 32, "num_blocks": 4},   # this is the baseline config
]

SEARCH_EPOCHS = 8   # short training budget per candidate -- this is what
                     # makes it "lightweight" instead of full NAS


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
best_score = -1
best_model = None
best_config = None

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

    # Simple accuracy-per-parameter efficiency score. Tune the weighting to
    # match how much you care about size vs accuracy for your report.
    efficiency_score = val_acc - 0.05 * (num_params / 50000)

    results.append({
        "base_filters": config["base_filters"],
        "num_blocks": config["num_blocks"],
        "val_accuracy": val_acc,
        "num_params": num_params,
        "efficiency_score": efficiency_score,
    })
    print(f"  val_acc={val_acc:.4f}  params={num_params}  score={efficiency_score:.4f}")

    if efficiency_score > best_score:
        best_score = efficiency_score
        best_config = config
        best_model = candidate

print(f"\nBest config from search: {best_config}")

with open("nas_search_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print("Saved nas_search_results.csv")

# retrain the winning architecture with the full training budget
print("\nRetraining winning architecture with full epochs...")
final_model = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES, **best_config)
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ],
)
arch_test_acc = final_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Architecture-optimized test accuracy: {arch_test_acc:.4f}  "
      f"(delta vs baseline: {arch_test_acc - baseline_acc:+.4f})")

final_model.save("arch_optimized_model.h5")
print("Saved arch_optimized_model.h5")

print("\nAll variants ready: baseline_model.h5, quantized_model.tflite, "
      "pruned_model.h5, arch_optimized_model.h5")
