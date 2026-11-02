/*
  TinyML Keyword Spotting -- ESP32 benchmark firmware
  ====================================================

  Chirale_TensorFlowLite version

  This sketch benchmarks the INT8 model variants plus the baseline FP32
  model exported by the Google Colab pipeline:

    0 = baseline INT8
    1 = post-training quantized INT8
    2 = pruned + INT8
    3 = architecture-optimized + INT8
    4 = baseline FP32

  The benchmark uses the 50 fixed Speech Commands test samples stored in
  test_samples.h. No microphone is required.

  For a fair comparison, change MODEL_VARIANT, upload, run once, and
  record the printed accuracy, latency, tensor-arena usage, and model size.

  Required Arduino library:
    Chirale_TensorFlowLite by Spazio Chirale

  Board:
    ESP32 Dev Module
*/

#include <Arduino.h>
#include <math.h>
#include <limits.h>

// Chirale's Arduino port of TensorFlow Lite Micro.
#include <Chirale_TensorFlowLite.h>

// TFLite Micro headers used by the Chirale examples/API.
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================
// MODEL SELECTION
// ============================================================
// Change ONLY this number, then Verify -> Upload for each model.
//   0 = baseline INT8
//   1 = quantized INT8
//   2 = pruned INT8
//   3 = architecture-optimized INT8
//   4 = baseline FP32
#define MODEL_VARIANT 0

#if MODEL_VARIANT == 0
  #include "model_baseline.h"
  #define MODEL_DATA g_baseline_model
  #define MODEL_DATA_LEN g_baseline_model_len
  const char* MODEL_NAME = "Baseline INT8";
#elif MODEL_VARIANT == 1
  #include "model_quantized.h"
  #define MODEL_DATA g_quantized_model
  #define MODEL_DATA_LEN g_quantized_model_len
  const char* MODEL_NAME = "Quantized INT8";
#elif MODEL_VARIANT == 2
  #include "model_pruned.h"
  #define MODEL_DATA g_pruned_model
  #define MODEL_DATA_LEN g_pruned_model_len
  const char* MODEL_NAME = "Pruned INT8";
#elif MODEL_VARIANT == 3
  #include "model_arch_optimized.h"
  #define MODEL_DATA g_arch_optimized_model
  #define MODEL_DATA_LEN g_arch_optimized_model_len
  const char* MODEL_NAME = "Architecture-optimized INT8";
#elif MODEL_VARIANT == 4
  #include "model_baseline_float32.h"
  #define MODEL_DATA g_baseline_model_fp32
  #define MODEL_DATA_LEN g_baseline_model_fp32_len
  const char* MODEL_NAME = "Baseline FP32";
#else
  #error "MODEL_VARIANT must be 0, 1, 2, 3, or 4"
#endif

#include "test_samples.h"

// ============================================================
// TENSOR ARENA
// ============================================================
// This is the scratch RAM available to TensorFlow Lite Micro.
// The actual requirement is reported with arena_used_bytes().
// 100 KB gives enough headroom while keeping the ESP32 well within
// its internal RAM capacity for this small keyword-spotting model.
constexpr size_t kTensorArenaSize = 90 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ============================================================
// TFLITE GLOBALS
// ============================================================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" TinyML KWS - ESP32 Benchmark");
  Serial.println("========================================");
  Serial.print("Model: ");
  Serial.println(MODEL_NAME);
  Serial.print("Model variant: ");
  Serial.println(MODEL_VARIANT);
  Serial.print("Model bytes: ");
  Serial.println(MODEL_DATA_LEN);
  Serial.println();

  // Load the FlatBuffer model directly from flash.
  model = tflite::GetModel(MODEL_DATA);

  if (model == nullptr) {
    Serial.println("ERROR: Could not load model.");
    return;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version = ");
    Serial.print(model->version());
    Serial.print(", library schema version = ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Register only the operators used by the DS-CNN model.
  // This avoids the large flash overhead of AllOpsResolver.
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddQuantize();

  // Chirale's example uses this four-argument MicroInterpreter form.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    Serial.println("Try increasing kTensorArenaSize.");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("ERROR: Input/output tensor is null.");
    return;
  }

  Serial.println("TensorFlow Lite Micro initialization: OK");

  // Actual tensor-arena memory used by this model.
  size_t arena_used = interpreter->arena_used_bytes();
  Serial.print("Tensor arena used (RAM): ");
  Serial.print(arena_used);
  Serial.println(" bytes");

  // Size of the model FlatBuffer stored in the generated .h file.
  Serial.print("Model binary size (flash): ");
  Serial.print(MODEL_DATA_LEN);
  Serial.println(" bytes");

  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);
  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);

  Serial.print("Output type: ");
  Serial.println(output->type);
  Serial.print("Output scale: ");
  Serial.println(output->params.scale, 8);
  Serial.print("Output zero point: ");
  Serial.println(output->params.zero_point);

  Serial.println();
  Serial.print("Running ");
  Serial.print(NUM_TEST_SAMPLES);
  Serial.println(" fixed Speech Commands test samples...");
  Serial.println();

  runBenchmark();
}

// ============================================================
// BENCHMARK
// ============================================================
void runBenchmark() {
  int correct = 0;
  unsigned long total_us = 0;
  unsigned long min_us = ULONG_MAX;
  unsigned long max_us = 0;

  // One timing value for every sample so standard deviation can be computed.
  unsigned long timings[NUM_TEST_SAMPLES];
  bool timing_valid[NUM_TEST_SAMPLES];

  for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
    timing_valid[i] = false;
  }

  const bool model_is_fp32 = (input->type == kTfLiteFloat32);

  const float input_scale = input->params.scale;
  const int input_zero_point = input->params.zero_point;
  const float output_scale = output->params.scale;
  const int output_zero_point = output->params.zero_point;

  if (!model_is_fp32 && input_scale == 0.0f) {
    Serial.println("ERROR: INT8 input quantization scale is zero.");
    return;
  }

  for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
    // ----------------------------------------------------------
    // Copy the MFCC sample into the model input.
    // INT8 models: quantize using that model's own scale/zero point.
    // FP32 model: copy the float values directly.
    // ----------------------------------------------------------
    for (int j = 0; j < FEATURE_SIZE; ++j) {
      float value = g_test_samples[i][j];

      if (model_is_fp32) {
        input->data.f[j] = value;
      } else {
        int32_t q = (int32_t)roundf(value / input_scale)
                    + input_zero_point;

        if (q < -128) q = -128;
        if (q > 127) q = 127;

        input->data.int8[j] = (int8_t)q;
      }
    }

    // ----------------------------------------------------------
    // Time ONLY the neural-network Invoke() call.
    // Input preparation and serial printing are excluded.
    // ----------------------------------------------------------
    unsigned long start_us = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long elapsed_us = micros() - start_us;

    if (invoke_status != kTfLiteOk) {
      Serial.print("Sample ");
      Serial.print(i);
      Serial.println(": ERROR - Invoke() failed");
      continue;
    }

    timings[i] = elapsed_us;
    timing_valid[i] = true;
    total_us += elapsed_us;

    if (elapsed_us < min_us) min_us = elapsed_us;
    if (elapsed_us > max_us) max_us = elapsed_us;

    // ----------------------------------------------------------
    // Argmax over the 10 output classes.
    // Support both INT8 and FP32 output tensors.
    // ----------------------------------------------------------
    int best_idx = 0;
    float best_score = -INFINITY;

    for (int c = 0; c < 10; ++c) {
      float score;

      if (model_is_fp32) {
        score = output->data.f[c];
      } else {
        int8_t raw = output->data.int8[c];
        score = (raw - output_zero_point) * output_scale;
      }

      if (score > best_score) {
        best_score = score;
        best_idx = c;
      }
    }

    bool is_correct = (best_idx == g_test_labels[i]);
    if (is_correct) correct++;

    Serial.print("Sample ");
    Serial.print(i);
    Serial.print(" | predicted=");
    Serial.print(kCategoryLabels[best_idx]);
    Serial.print(" | true=");
    Serial.print(kCategoryLabels[g_test_labels[i]]);
    Serial.print(" | score=");
    Serial.print(best_score, 5);
    Serial.print(" | latency_us=");
    Serial.print(elapsed_us);
    Serial.print(" | ");
    Serial.println(is_correct ? "CORRECT" : "WRONG");
  }

  // ------------------------------------------------------------
  // Calculate summary statistics using only successful invokes.
  // ------------------------------------------------------------
  int valid_count = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
    if (timing_valid[i]) valid_count++;
  }

  Serial.println();
  Serial.println("========================================");
  Serial.println(" BENCHMARK SUMMARY");
  Serial.println("========================================");

  Serial.print("Model: ");
  Serial.println(MODEL_NAME);

  Serial.print("Test accuracy: ");
  Serial.print(correct);
  Serial.print("/");
  Serial.print(NUM_TEST_SAMPLES);
  Serial.print(" = ");
  Serial.print(100.0f * correct / NUM_TEST_SAMPLES, 2);
  Serial.println("%");

  if (valid_count > 0) {
    float mean_us = (float)total_us / valid_count;

    float sq_sum = 0.0f;
    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
      if (!timing_valid[i]) continue;
      float diff = (float)timings[i] - mean_us;
      sq_sum += diff * diff;
    }
    float stddev_us = sqrtf(sq_sum / valid_count);

    Serial.print("Mean latency (us): ");
    Serial.println(mean_us, 2);

    Serial.print("Min latency (us): ");
    Serial.println(min_us);

    Serial.print("Max latency (us): ");
    Serial.println(max_us);

    Serial.print("Stddev latency (us): ");
    Serial.println(stddev_us, 2);
  } else {
    Serial.println("No successful inference timings recorded.");
  }

  Serial.print("Tensor arena used / RAM (bytes): ");
  Serial.println(interpreter->arena_used_bytes());

  Serial.print("Model binary / model flash (bytes): ");
  Serial.println(MODEL_DATA_LEN);

  Serial.println("========================================");
  Serial.println("Benchmark finished.");
  Serial.println("========================================");
}

void loop() {
  // Run once after boot. The same flashed firmware should not rerun the
  // benchmark automatically because repeated runs would complicate logging.
  delay(1000);
}
