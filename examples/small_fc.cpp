#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/array/jit/jit.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>
#include <dali/utils/make_message.h>
#include <dali/utils/timer.h>

#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");
DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_int32(batch_size, 256, "Batch size");


int main (int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "MNIST training using simple convnet\n"
        "------------------------------------\n"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_device == -1) {
        default_preferred_device = Device::cpu();
    }
#ifdef DALI_USE_CUDA
    if (FLAGS_device >= 0) {
        ASSERT2(FLAGS_device < Device::num_gpus(), utils::make_message("Cannot run on GPU ", FLAGS_device, ", only found ", Device::num_gpus(), " gpus."));
        default_preferred_device = Device::gpu(FLAGS_device);
    }
#endif
    std::cout << "Running on " << default_preferred_device.description() << "." << std::endl;
    utils::random::set_seed(123123);
    WithCudnnPreference cudnn_pref(FLAGS_use_cudnn);
    op::jit::WithJITFusionPreference jit_pref(FLAGS_use_jit_fusion);
    std::cout << "Use CuDNN = " << (cudnn_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "Use JIT Fusion = " << (op::jit::jit_fusion_preference() ? "True" : "False") << "." << std::endl;

    auto a = op::uniform(-20.0f, 20.0f, {2, 5});
    // auto scaler = Array::zeros({2, 5}, DTYPE_FLOAT);
    // auto bias = Array::zeros({1, 5}, DTYPE_FLOAT);
    // auto mean_a = a.mean({-1}, true);
    // auto a_stdev = op::sqrt(op::square(a - mean_a) / Array(a.shape()[1]));
    // auto a_rescaled = (a - mean_a) * ((scaler + 1.0) / a_stdev) + bias;
    auto b = op::uniform(-20.0f, 20.0f, {2, 5});
    auto c = a.transpose() + b.transpose();
    c.print();
    // auto exped = op::exp(a - op::max(a, {-1}, true));
    // auto fused_softmax = exped / op::sum(exped, {-1}, true);
    // fused_softmax.eval();
    // fused_softmax.print();


}
