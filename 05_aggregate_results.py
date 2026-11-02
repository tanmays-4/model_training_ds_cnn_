"""
Stage 5 -- Aggregate ESP32 Serial output + combined_results_summary.csv into
the final report table and Pareto plots.

HOW TO GET THE INPUT FILE:
  1. Flash esp32_firmware_main.ino, open Serial Monitor (115200 baud).
  2. After "DONE" prints, copy everything from the CSV header line down to
     (not including) "DONE" into a text file: esp32_serial_log.csv
     (Arduino IDE: Serial Monitor has a "Copy" / "Save" option; or use
     `arduino-cli monitor` piped to a file, or PuTTY logging.)
  3. If you add energy measurements separately (INA219 log timestamped
     against the GPIO trigger), merge a `energy_uj_per_inference` column
     into esp32_serial_log.csv by hand or extend this script.

Run: python 05_aggregate_results.py
Needs in the same folder: combined_results_summary.csv, esp32_serial_log.csv
Produces: final_report_table.csv, pareto_accuracy_vs_latency.png,
          pareto_accuracy_vs_memory.png
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCURACY_SOURCE_CSV = "combined_results_summary.csv"   # from 04_combined_variants.py
DEVICE_CSV = "esp32_serial_log.csv"                     # from the ESP32 firmware

NAME_MAP = {
    # combined_results_summary.csv "config" value -> firmware "model" value
    "baseline_fp32": "baseline_fp32",
    "baseline_int8": "baseline_int8",
    "pruned_int8": "pruned_int8",
    "nas_v2_int8": "nas_v2_int8",
    "nas_v2_pruned_int8": "nas_v2_pruned_int8",
}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    acc_rows = {r["config"]: r for r in load_csv(ACCURACY_SOURCE_CSV)}
    try:
        dev_rows = {r["model"]: r for r in load_csv(DEVICE_CSV)}
    except FileNotFoundError:
        print(f"{DEVICE_CSV} not found yet -- run the ESP32 firmware first "
              f"and save its Serial output there. Printing accuracy-only "
              f"table for now.")
        dev_rows = {}

    merged = []
    for config, firmware_name in NAME_MAP.items():
        acc_row = acc_rows.get(config)
        if acc_row is None:
            continue
        dev_row = dev_rows.get(firmware_name, {})
        merged.append({
            "config": config,
            "accuracy_offline": acc_row["accuracy"],
            "accuracy_on_device": dev_row.get("accuracy", ""),
            "params": acc_row["params"],
            "tflite_bytes": acc_row["tflite_bytes"],
            "mean_latency_us": dev_row.get("mean_latency_us", ""),
            "p95_latency_us": dev_row.get("p95_latency_us", ""),
            "arena_used_bytes": dev_row.get("arena_used_bytes", ""),
        })

    with open("final_report_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    print("Wrote final_report_table.csv -- this is your headline results table.")

    if not dev_rows:
        print("Skipping plots (no device data yet).")
        return

    # --- Pareto plot: accuracy vs latency ---
    plt.figure(figsize=(6, 5))
    for row in merged:
        if row["mean_latency_us"] == "":
            continue
        x = float(row["mean_latency_us"])
        y = float(row["accuracy_on_device"] or row["accuracy_offline"])
        plt.scatter(x, y, s=60)
        plt.annotate(row["config"], (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    plt.xlabel("Mean inference latency (µs)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Latency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_accuracy_vs_latency.png", dpi=150)
    print("Wrote pareto_accuracy_vs_latency.png")

    # --- Pareto plot: accuracy vs memory (tensor arena) ---
    plt.figure(figsize=(6, 5))
    for row in merged:
        if row["arena_used_bytes"] == "":
            continue
        x = float(row["arena_used_bytes"])
        y = float(row["accuracy_on_device"] or row["accuracy_offline"])
        plt.scatter(x, y, s=60)
        plt.annotate(row["config"], (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    plt.xlabel("Tensor arena RAM usage (bytes)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Memory (RAM)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_accuracy_vs_memory.png", dpi=150)
    print("Wrote pareto_accuracy_vs_memory.png")


if __name__ == "__main__":
    main()
