"""
Stage 1 — Data preparation + baseline DS-CNN training
Run this in Google Colab (needs internet access; GPU runtime recommended).

Pipeline:
  Download Google Speech Commands v0.02
  -> leak-free train/val/test split (using the dataset's own speaker-based lists)
  -> extract MFCC features (40 coeffs x 49 frames, matching the "Hello Edge" KWS paper shape)
  -> train a small Depthwise-Separable CNN (DS-CNN) baseline
  -> save baseline_model.h5 (Keras) for later compression + conversion stages

Classes: yes, no, up, down, stop, go, left, right, unknown, silence  (10 total)
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # keep this consistent with steps 2/3,
                                            # which need legacy Keras for
                                            # tensorflow_model_optimization.
                                            # Must come before `import tensorflow`.

import glob
import random
import tarfile
import urllib.request

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 320          # 16000 * 1.0 / 320 ~= 50 frames -> we pad/crop to 49
N_FRAMES = 49

TARGET_WORDS = ["yes", "no", "up", "down", "stop", "go", "left", "right"]
CLASSES = TARGET_WORDS + ["unknown", "silence"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MAX_UNKNOWN_PER_SPLIT = 4000   # cap so "unknown" doesn't dominate the dataset
MAX_SILENCE_PER_SPLIT = 2000

DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATA_DIR = "speech_commands_data"
ARCHIVE_PATH = "speech_commands_v0.02.tar.gz"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Step 1: Download + extract
# ---------------------------------------------------------------------------
def download_and_extract():
    if os.path.isdir(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"Dataset already present at {DATA_DIR}, skipping download.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading Speech Commands v0.02 (~2.3GB)... this can take a while.")
    urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
    print("Extracting...")
    with tarfile.open(ARCHIVE_PATH) as tar:
        tar.extractall(DATA_DIR)
    print("Done.")


# ---------------------------------------------------------------------------
# Step 2: Build a leak-free file list using the dataset's own split lists.
# The dataset ships validation_list.txt and testing_list.txt containing
# relative paths of files that must go to val/test. Everything else is train.
# This guarantees the same speaker never appears in two different splits.
# ---------------------------------------------------------------------------
def build_file_lists():
    val_list_path = os.path.join(DATA_DIR, "validation_list.txt")
    test_list_path = os.path.join(DATA_DIR, "testing_list.txt")

    with open(val_list_path) as f:
        val_set = set(line.strip() for line in f if line.strip())
    with open(test_list_path) as f:
        test_set = set(line.strip() for line in f if line.strip())

    all_word_dirs = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith("_")
    ]

    train_files, val_files, test_files = [], [], []

    for word in all_word_dirs:
        label = word if word in TARGET_WORDS else "unknown"
        wav_paths = glob.glob(os.path.join(DATA_DIR, word, "*.wav"))
        for wav_path in wav_paths:
            rel_path = os.path.join(word, os.path.basename(wav_path))
            if rel_path in val_set:
                val_files.append((wav_path, label))
            elif rel_path in test_set:
                test_files.append((wav_path, label))
            else:
                train_files.append((wav_path, label))

    def cap_unknown(file_list, max_per_split):
        known = [x for x in file_list if x[1] != "unknown"]
        unknown = [x for x in file_list if x[1] == "unknown"]
        random.shuffle(unknown)
        return known + unknown[:max_per_split]

    train_files = cap_unknown(train_files, MAX_UNKNOWN_PER_SPLIT)
    val_files = cap_unknown(val_files, MAX_UNKNOWN_PER_SPLIT // 4)
    test_files = cap_unknown(test_files, MAX_UNKNOWN_PER_SPLIT // 4)

    print(f"train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")
    return train_files, val_files, test_files


# ---------------------------------------------------------------------------
# Step 3: "silence" class — random 1s crops of background noise clips
# ---------------------------------------------------------------------------
def generate_silence_entries(n_per_split):
    noise_dir = os.path.join(DATA_DIR, "_background_noise_")
    noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    entries = []
    for _ in range(n_per_split):
        noise_path = random.choice(noise_files)
        entries.append((noise_path, "silence"))
    return entries


# ---------------------------------------------------------------------------
# Step 4: Audio -> MFCC feature extraction
# ---------------------------------------------------------------------------
def load_audio_fixed_length(path, is_silence=False):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if is_silence:
        # random 1-second crop from a (usually much longer) noise clip
        if len(audio) > N_SAMPLES:
            start = random.randint(0, len(audio) - N_SAMPLES)
            audio = audio[start:start + N_SAMPLES]
        else:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        audio = audio * random.uniform(0.1, 0.5)  # vary loudness
        return audio

    # pad or trim to exactly 1 second
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )  # shape: (N_MFCC, frames)

    # pad/crop time axis to exactly N_FRAMES
    if mfcc.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :N_FRAMES]

    # per-sample normalization (zero mean, unit variance)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return mfcc.T  # shape: (N_FRAMES, N_MFCC) -> time-major, easier for Conv2D


def build_feature_dataset(file_list):
    X = np.zeros((len(file_list), N_FRAMES, N_MFCC, 1), dtype=np.float32)
    y = np.zeros((len(file_list),), dtype=np.int64)

    for i, (path, label) in enumerate(file_list):
        is_silence = label == "silence"
        audio = load_audio_fixed_length(path, is_silence=is_silence)
        mfcc = extract_mfcc(audio)
        X[i, :, :, 0] = mfcc
        y[i] = LABEL_TO_IDX[label]

        if i % 500 == 0:
            print(f"  processed {i}/{len(file_list)}")

    return X, y


# ---------------------------------------------------------------------------
# Step 5: Baseline DS-CNN model
# ---------------------------------------------------------------------------
def ds_conv_block(x, filters, strides=(1, 1)):
    """One depthwise-separable conv block: depthwise conv -> BN -> ReLU
    -> pointwise (1x1) conv -> BN -> ReLU."""
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=strides, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_ds_cnn(input_shape, num_classes, base_filters=32, num_blocks=4):
    inputs = layers.Input(shape=input_shape)

    # standard conv "stem" to lift channel count before depthwise blocks
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

    return models.Model(inputs, outputs, name="ds_cnn_baseline")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if os.path.exists("features.npz"):
        print("Found existing features.npz -- reusing it instead of "
              "recomputing MFCCs (delete the file first if you want a "
              "fresh extraction, e.g. after changing MFCC settings above).")
        cached = np.load("features.npz")
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_val, y_val = cached["X_val"], cached["y_val"]
        X_test, y_test = cached["X_test"], cached["y_test"]
    else:
        download_and_extract()
        train_files, val_files, test_files = build_file_lists()

        train_files += generate_silence_entries(MAX_SILENCE_PER_SPLIT)
        val_files += generate_silence_entries(MAX_SILENCE_PER_SPLIT // 4)
        test_files += generate_silence_entries(MAX_SILENCE_PER_SPLIT // 4)

        random.shuffle(train_files)

        print("Extracting MFCC features (train)...")
        X_train, y_train = build_feature_dataset(train_files)
        print("Extracting MFCC features (val)...")
        X_val, y_val = build_feature_dataset(val_files)
        print("Extracting MFCC features (test)...")
        X_test, y_test = build_feature_dataset(test_files)

        np.savez_compressed(
            "features.npz",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )
        print("Saved features.npz (reused by later stages so you don't recompute MFCCs)")

    model = build_ds_cnn(
        input_shape=(N_FRAMES, N_MFCC, 1),
        num_classes=len(CLASSES),
        base_filters=32,
        num_blocks=4,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Baseline test accuracy: {test_acc:.4f}")

    model.save("baseline_model.h5")
    print("Saved baseline_model.h5")


if __name__ == "__main__":
    main()
