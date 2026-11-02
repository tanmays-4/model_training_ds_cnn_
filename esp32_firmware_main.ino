/*
 * ESP32-WROOM-32 benchmark firmware for the TinyML compression paper.
 *
 * Loops through every deployed model variant (baseline INT8, baseline FP32
 * reference, pruned INT8, NAS-v2 INT8, NAS-v2+Pruned INT8), runs all
 * NUM_TEST_SAMPLES embedded test samples through each, and prints one CSV
 * line per model to Serial with: accuracy, mean/p95 latency, and tensor
 * arena (RAM) usage. Pipe the Serial output into a text file and feed it
 * to 05_aggregate_results.py.
 *
 * SETUP
 * -----
 * 1. Arduino IDE, board = "ESP32 Dev Module" (or your WROOM-32 variant),
 *    partition scheme with enough flash for ~5 models (Default 4MB is fine
 *    for models in the tens-of-KB range you're working with).
 * 2. Install a TensorFlow Lite Micro Arduino library, e.g. one of:
 *      - "Chirale_TensorFlowLite" (actively maintained ESP32 port), or
 *      - "TensorFlowLite_ESP32"
 *    Both expose the same core tflite-micro C++ API used below
 *    (tflite::MicroInterpreter, MicroMutableOpResolver, etc.). If your
 *    installed library's header paths differ slightly from the #includes
 *    below, adjust them to match -- the API itself is stable across ports.
 * 3. Copy the whole esp32_firmware/ folder (all model_*.h headers +
 *    test_samples.h, produced by stage 3 and stage 4) into this sketch's
 *    folder so the #includes below resolve.
 * 4. Flash, open Serial Monitor at 115200 baud.
 *
 * ENERGY MEASUREMENT HOOK
 * ------------------------
 * GPIO_ENERGY_TRIGGER toggles HIGH immediately before each inference batch
 * and LOW immediately after, so an external INA219/INA226 logger (on a
 * second MCU) or a scope/power-meter can be triggered on that edge to
 * capture current draw during inference. If you don't have that hardware
 * yet, the pin toggle is harmless to leave in -- just ignore it.
 */

#include <Arduino.h>

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ---- Model headers (from esp32_firmware/, produced by stages 3 and 4) ----
#include "model_baseline.h"            // g_baseline_model            (INT8)
#include "model_baseline_float32.h"    // g_baseline_model_fp32       (FP32, reference only)
#include "model_pruned.h"              // g_pruned_model              (INT8)
#include "model_nas_v2.h"              // g_nas_v2_model              (INT8)
#include "model_nas_pruned.h"          // g_nas_pruned_model          (INT8)
#include "test_samples.h"              // g_test_samples, g_test_labels, kCategoryLabels

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define GPIO_ENERGY_TRIGGER 2          // change to a free GPIO on your board
constexpr int kTensorArenaSize = 80 * 1024;   // generous; actual usage printed per model
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

struct ModelVariant {
  const char* name;
  const unsigned char* data;
  unsigned int len;
  bool is_int8;   // false only for the fp32 reference model
};

static ModelVariant kVariants[] = {
  {"baseline_fp32",    g_baseline_model_fp32,  g_baseline_model_fp32_len,  false},
  {"baseline_int8",    g_baseline_model,        g_baseline_model_len,       true},
  {"pruned_int8",      g_pruned_model,          g_pruned_model_len,         true},
  {"nas_v2_int8",      g_nas_v2_model,          g_nas_v2_model_len,         true},
  {"nas_v2_pruned_int8", g_nas_pruned_model,    g_nas_pruned_model_len,     true},
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

// ---------------------------------------------------------------------------
// Benchmark one model variant: run all test samples, return via Serial.
// ---------------------------------------------------------------------------
void benchmarkVariant(const ModelVariant& variant) {
  const tflite::Model* model = tflite::GetModel(variant.data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR: %s schema version mismatch\n", variant.name);
    return;
  }

  // Register exactly the ops your DS-CNN graph uses. If you change the
  // architecture (e.g. add new op types), add them here -- an unresolved
  // op fails interpreter.AllocateTensors() with a clear error on Serial.
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddMean();          // GlobalAveragePooling2D lowers to MEAN
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  // AddLogistic / others only if your graph needs them.

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                        kTensorArenaSize);
  TfLiteStatus alloc_status = interpreter.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.printf("ERROR: %s AllocateTensors failed (arena too small? "
                  "increase kTensorArenaSize)\n", variant.name);
    return;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  float in_scale = 1.0f;
  int32_t in_zero_point = 0;
  if (variant.is_int8) {
    in_scale = input->params.scale;
    in_zero_point = input->params.zero_point;
  }

  int correct = 0;
  unsigned long latencies_us[NUM_TEST_SAMPLES];

  digitalWrite(GPIO_ENERGY_TRIGGER, HIGH);   // energy logger: batch start

  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    // Fill input tensor from the shared float32 test sample, quantizing
    // on-device with THIS model's own scale/zero-point (each variant was
    // quantized independently in Python, so scales differ between models).
    if (variant.is_int8) {
      int8_t* in_data = input->data.int8;
      for (int j = 0; j < FEATURE_SIZE; j++) {
        float v = g_test_samples[i][j];
        int32_t q = (int32_t)roundf(v / in_scale) + in_zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        in_data[j] = (int8_t)q;
      }
    } else {
      float* in_data = input->data.f;
      for (int j = 0; j < FEATURE_SIZE; j++) {
        in_data[j] = g_test_samples[i][j];
      }
    }

    unsigned long t0 = micros();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long t1 = micros();
    latencies_us[i] = t1 - t0;

    if (invoke_status != kTfLiteOk) {
      Serial.printf("ERROR: %s invoke failed on sample %d\n", variant.name, i);
      continue;
    }

    // Argmax over output (int8 or float, same ordering either way)
    int pred = 0;
    if (variant.is_int8) {
      int8_t best = output->data.int8[0];
      for (int c = 1; c < output->dims->data[1]; c++) {
        if (output->data.int8[c] > best) { best = output->data.int8[c]; pred = c; }
      }
    } else {
      float best = output->data.f[0];
      for (int c = 1; c < output->dims->data[1]; c++) {
        if (output->data.f[c] > best) { best = output->data.f[c]; pred = c; }
      }
    }
    if (pred == g_test_labels[i]) correct++;
  }

  digitalWrite(GPIO_ENERGY_TRIGGER, LOW);    // energy logger: batch end

  // Stats
  unsigned long sum = 0, max_lat = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    sum += latencies_us[i];
    if (latencies_us[i] > max_lat) max_lat = latencies_us[i];
  }
  float mean_us = (float)sum / NUM_TEST_SAMPLES;

  // simple p95 via sort (NUM_TEST_SAMPLES is small, e.g. 50)
  unsigned long sorted[NUM_TEST_SAMPLES];
  memcpy(sorted, latencies_us, sizeof(sorted));
  for (int i = 0; i < NUM_TEST_SAMPLES; i++)
    for (int j = i + 1; j < NUM_TEST_SAMPLES; j++)
      if (sorted[j] < sorted[i]) { unsigned long t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
  unsigned long p95_us = sorted[(int)(0.95f * (NUM_TEST_SAMPLES - 1))];

  float accuracy = (float)correct / NUM_TEST_SAMPLES;
  size_t arena_used = interpreter.arena_used_bytes();

  // CSV line: model,accuracy,mean_latency_us,p95_latency_us,max_latency_us,arena_used_bytes,model_flash_bytes
  Serial.printf("%s,%.4f,%.1f,%lu,%lu,%u,%u\n",
                variant.name, accuracy, mean_us, p95_us, max_lat,
                (unsigned)arena_used, variant.len);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  pinMode(GPIO_ENERGY_TRIGGER, OUTPUT);
  digitalWrite(GPIO_ENERGY_TRIGGER, LOW);
  delay(2000);

  Serial.println("model,accuracy,mean_latency_us,p95_latency_us,max_latency_us,arena_used_bytes,model_flash_bytes");

  for (int i = 0; i < kNumVariants; i++) {
    benchmarkVariant(kVariants[i]);
    delay(200);
  }

  Serial.println("DONE");
}

void loop() {
  // nothing -- benchmark runs once in setup(). Reset the board to re-run.
}
