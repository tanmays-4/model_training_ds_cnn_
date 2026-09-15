/*
  TinyML Keyword Spotting -- ESP32 benchmarking firmware
  ========================================================

  This sketch loads ONE model variant at a time (chosen below by which
  header you #include and which byte array you point kModelData at),
  runs inference on a fixed set of test samples, and prints per-sample
  latency plus tensor arena (RAM) usage over Serial.

  To benchmark all four variants: change the MODEL_VARIANT define below,
  reflash, and record the printed results. Repeat for each variant.
  Keeping everything else identical is what makes the comparison fair.

  REQUIRED LIBRARY:
    Install "esp-tflite-micro" via Arduino Library Manager
    (Espressif's actively-maintained TFLite Micro port for ESP32).
    Board package: esp32 by Espressif Systems (Boards Manager).

  WIRING: none required for this benchmarking sketch -- it runs on
  pre-recorded test samples baked into flash (test_samples.h), so no
  microphone is needed yet.
*/

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------
// STEP 1: pick which variant to benchmark by uncommenting ONE line below.
// Only one at a time -- flash, run, record results, then swap.
// ---------------------------------------------------------------------
#include "model_baseline.h"
// #include "model_quantized.h"
// #include "model_pruned.h"
// #include "model_arch_optimized.h"
// #include "model_baseline_float32.h"   // NOTE: float32 model needs a much
                                          // larger tensor arena below, and a
                                          // different ops resolver setup --
                                          // see comment near kTensorArenaSize.

#include "test_samples.h"

// Match this to whichever #include you enabled above.
#define MODEL_DATA g_baseline_model
#define MODEL_DATA_LEN g_baseline_model_len

// ---------------------------------------------------------------------
// STEP 2: tensor arena -- the scratch RAM TFLM uses for activations.
// Start conservative and increase if AllocateTensors() fails; this
// final chosen value IS one of your benchmark numbers (RAM usage).
// ---------------------------------------------------------------------
constexpr int kTensorArenaSize = 60 * 1024;   // 60 KB, typical starting point
                                                // for a small DS-CNN at INT8
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== TinyML KWS ESP32 Benchmark ===");

  model = tflite::GetModel(MODEL_DATA);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: model schema version mismatch");
    return;
  }

  // Op resolver: list every op your model actually uses. Keeping this
  // minimal (rather than using AllOpsResolver) saves flash -- worth
  // reporting as part of your flash-size comparison.
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddMean();          // used by GlobalAveragePooling2D
  resolver.AddReshape();
  resolver.AddQuantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed -- increase kTensorArenaSize");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  size_t used_bytes = interpreter->arena_used_bytes();
  Serial.print("Tensor arena used: ");
  Serial.print(used_bytes);
  Serial.println(" bytes  <-- this is your RAM benchmark number");

  Serial.print("Model size (flash): ");
  Serial.print(MODEL_DATA_LEN);
  Serial.println(" bytes  <-- this is your flash/model-size benchmark number");

  run_benchmark();
}

void run_benchmark() {
  int correct = 0;
  unsigned long total_us = 0;
  unsigned long min_us = ULONG_MAX;
  unsigned long max_us = 0;
  unsigned long timings[NUM_TEST_SAMPLES];

  float in_scale = input->params.scale;
  int in_zero_point = input->params.zero_point;
  float out_scale = output->params.scale;
  int out_zero_point = output->params.zero_point;

  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    // Quantize this sample using THIS model's own scale/zero-point --
    // necessary because each variant gets its own quantization params
    // from the converter.
    for (int j = 0; j < FEATURE_SIZE; j++) {
      float v = g_test_samples[i][j];
      int32_t q = (int32_t)round(v / in_scale) + in_zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input->data.int8[j] = (int8_t)q;
    }

    unsigned long start_us = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long elapsed_us = micros() - start_us;

    if (invoke_status != kTfLiteOk) {
      Serial.println("ERROR: Invoke() failed");
      continue;
    }

    timings[i] = elapsed_us;
    total_us += elapsed_us;
    if (elapsed_us < min_us) min_us = elapsed_us;
    if (elapsed_us > max_us) max_us = elapsed_us;

    // dequantize output, find argmax
    int best_idx = 0;
    float best_val = -1e9;
    for (int c = 0; c < 10; c++) {
      float val = (output->data.int8[c] - out_zero_point) * out_scale;
      if (val > best_val) {
        best_val = val;
        best_idx = c;
      }
    }

    bool is_correct = (best_idx == g_test_labels[i]);
    if (is_correct) correct++;

    Serial.print("Sample ");
    Serial.print(i);
    Serial.print(": predicted=");
    Serial.print(kCategoryLabels[best_idx]);
    Serial.print(" true=");
    Serial.print(kCategoryLabels[g_test_labels[i]]);
    Serial.print(" latency_us=");
    Serial.println(elapsed_us);
  }

  float mean_us = (float)total_us / NUM_TEST_SAMPLES;

  // standard deviation
  float sq_sum = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    float diff = timings[i] - mean_us;
    sq_sum += diff * diff;
  }
  float stddev_us = sqrt(sq_sum / NUM_TEST_SAMPLES);

  Serial.println("\n=== Benchmark summary ===");
  Serial.print("Accuracy on test samples: ");
  Serial.print(correct);
  Serial.print("/");
  Serial.println(NUM_TEST_SAMPLES);
  Serial.print("Mean latency (us): ");
  Serial.println(mean_us);
  Serial.print("Min latency (us): ");
  Serial.println(min_us);
  Serial.print("Max latency (us): ");
  Serial.println(max_us);
  Serial.print("Stddev latency (us): ");
  Serial.println(stddev_us);
}

void loop() {
  // benchmark runs once in setup(); nothing to repeat here
  delay(10000);
}
