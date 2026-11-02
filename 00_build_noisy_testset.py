"""
Stage 0b -- Build a synthetic "noisy" test set for robustness/generalization
results, using ONLY data you already have (no new downloads, no new
recordings). This is dataset #3 for your report:

  1. Speech Commands v0.02  (clean, in-distribution)   <- features.npz
  2. Speech Commands v0.01  (cross-version check)       <- see note below
  3. THIS SCRIPT: v0.02 test set with background noise mixed in at
     controlled SNR (10dB and 0dB) -- tests whether your compressed models
     degrade faster than the baseline under realistic deployment noise.
     This is exactly the kind of "deployed vs lab accuracy" robustness
     result that strengthens an empirical benchmarking paper.

Run this AFTER 01_data_and_baseline.py (needs features.npz and the original
raw .wav files still present in speech_commands_data/, since we remix from
the original audio, not the already-extracted MFCCs).

Outputs:
  noisy_test_features_10db.npz
  noisy_test_features_0db.npz

To get Speech Commands v0.01 as dataset #2: re-run 01_data_and_baseline.py
with DATA_URL changed to
  http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
and DATA_DIR changed to a different folder (e.g. "speech_commands_data_v01"),
then evaluate your already-trained models against ITS test split the same
way this script evaluates against the noisy set. Do not retrain on v0.01 --
the point is to test whether your v0.02-trained models generalize to a
different recording session of the same task.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob
import random
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 320
N_FRAMES = 49

TARGET_WORDS = ["yes", "no", "up", "down", "stop", "go", "left", "right"]
CLASSES = TARGET_WORDS + ["unknown", "silence"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_DIR = "speech_commands_data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_at_snr(clean, noise, snr_db):
    """Scale `noise` so the mix hits the target SNR (dB) against `clean`,
    then add. Both are assumed same length."""
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / (noise_rms + 1e-12)
    mixed = clean + noise * scale
    # prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def load_audio_fixed_length(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :N_FRAMES]
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return mfcc.T


def build_test_file_list():
    """Rebuilds the same test-file list logic as stage 1, so the noisy set
    lines up with the same held-out speakers (no leakage)."""
    test_list_path = os.path.join(DATA_DIR, "testing_list.txt")
    with open(test_list_path) as f:
        test_set = set(line.strip() for line in f if line.strip())

    all_word_dirs = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith("_")
    ]

    test_files = []
    for word in all_word_dirs:
        label = word if word in TARGET_WORDS else "unknown"
        for wav_path in glob.glob(os.path.join(DATA_DIR, word, "*.wav")):
            rel_path = os.path.join(word, os.path.basename(wav_path))
            if rel_path in test_set:
                test_files.append((wav_path, label))

    known = [x for x in test_files if x[1] != "unknown"]
    unknown = [x for x in test_files if x[1] == "unknown"]
    random.shuffle(unknown)
    test_files = known + unknown[:1000]   # cap, matches stage 1's proportions
    return test_files


def build_noisy_dataset(test_files, noise_files, snr_db):
    X = np.zeros((len(test_files), N_FRAMES, N_MFCC, 1), dtype=np.float32)
    y = np.zeros((len(test_files),), dtype=np.int64)

    for i, (path, label) in enumerate(test_files):
        clean = load_audio_fixed_length(path)
        noise_path = random.choice(noise_files)
        noise_full, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
        if len(noise_full) > N_SAMPLES:
            start = random.randint(0, len(noise_full) - N_SAMPLES)
            noise = noise_full[start:start + N_SAMPLES]
        else:
            noise = np.pad(noise_full, (0, N_SAMPLES - len(noise_full)))

        mixed = clean if snr_db is None else mix_at_snr(clean, noise, snr_db)
        mfcc = extract_mfcc(mixed)
        X[i, :, :, 0] = mfcc
        y[i] = LABEL_TO_IDX[label]

        if i % 500 == 0:
            print(f"  [{snr_db}dB] processed {i}/{len(test_files)}")

    return X, y


def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"{DATA_DIR} not found -- run 01_data_and_baseline.py first "
            f"(and don't delete the extracted speech_commands_data/ folder)."
        )

    noise_dir = os.path.join(DATA_DIR, "_background_noise_")
    noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    if not noise_files:
        raise SystemExit(f"No background noise files found in {noise_dir}")

    test_files = build_test_file_list()
    print(f"Building noisy test sets from {len(test_files)} clean test files "
          f"and {len(noise_files)} noise clips.")

    # None = the SAME file list with no noise added. This is the fair clean
    # reference for the noisy sets (their class mix differs from features.npz's
    # test split, e.g. no "silence" class), so compare noisy vs THIS, not vs
    # the headline test accuracy.
    for snr_db in (None, 10, 0):
        X, y = build_noisy_dataset(test_files, noise_files, snr_db)
        out_path = ("clean_test_features.npz" if snr_db is None
                    else f"noisy_test_features_{snr_db}db.npz")
        np.savez_compressed(out_path, X_test=X, y_test=y)
        print(f"Saved {out_path}  ({len(X)} samples)")

    print("\nEvaluate each of your saved models (baseline, pruned, nas_v2, "
          "nas_pruned -- fp32 .h5 or quantized .tflite) against these two "
          "npz files the same way stage 2/4 evaluate against X_test/y_test. "
          "Report accuracy-vs-SNR curves per model as your robustness result.")


if __name__ == "__main__":
    main()
