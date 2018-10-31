#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/tensor/op/spatial.h>
#include <dali/tensor/layers/conv.h>
#include <dali/array/jit/jit.h>
#include <dali/array/expression/computation.h>
#include <dali/array/memory/synchronized_memory.h>
#include <dali/array/memory/memory_bank.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>
#include <dali/utils/make_message.h>
#include <dali/utils/timer.h>

#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");
DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_bool(use_nchw, true, "Whether to use NCHW or NHWC.");
DEFINE_string(path, utils::dir_join({STR(DALI_EXAMPLES_DATA_DIR), "mnist"}), "Location of mnist data");
DEFINE_int32(batch_size, 256, "Batch size");
DEFINE_int32(epochs, 2, "Epochs");

struct MnistCnn {
    ConvLayer conv1;
    ConvLayer conv2;
    Layer     fc1;
    Layer     fc2;
    bool use_nchw_;

    MnistCnn(bool use_nchw) : use_nchw_(use_nchw) {
        conv1 = ConvLayer(32, 1, 5, 5, 1, 1, PADDING_T_SAME, use_nchw ? "NCHW" : "NHWC");
        conv2 = ConvLayer(64, 32, 5, 5, 1, 1, PADDING_T_SAME, use_nchw ? "NCHW" : "NHWC");
        fc1 = Layer(7 * 7 * 64, 1024);
        fc2 = Layer(1024, 10);
    }

    Tensor activate(Tensor images, float keep_prob) const {
        images = images.reshape(use_nchw_ ? shape_t{-1, 1, 28, 28} : shape_t{-1, 28, 28, 1});
        // shape (B, 1, 28, 28)
        Tensor out = conv1.activate(images).relu();
        out = tensor_ops::max_pool(out, 2, 2, -1, -1, PADDING_T_VALID, use_nchw_ ? "NCHW" : "NHWC");
        // shape (B, 32, 14, 14)
        out = conv2.activate(out).relu();
        out = tensor_ops::max_pool(out, 2, 2, -1, -1, PADDING_T_VALID, use_nchw_ ? "NCHW" : "NHWC");
        // shape (B, 64, 7, 7)
        out = out.reshape({out.shape()[0], 7 * 7 * 64});
        out = fc1.activate(out).relu();
        // shape (B, 1024)
        out = tensor_ops::dropout(out, 1.0 - keep_prob);
        out = fc2.activate(out);
        // shape (B, 10)
        return out;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({conv1.parameters(),
                                   conv2.parameters(),
                                   fc1.parameters(),
                                   fc2.parameters()
                               });
    }
};

double accuracy(const MnistCnn& model, Tensor images, Tensor labels, int batch_size) {
    graph::NoBackprop nb;
    int num_images = images.shape()[0].value();
    auto num_correct = Array::zeros({}, DTYPE_INT32);
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        Slice batch_slice(batch_start, std::min(batch_start + batch_size, num_images));
        auto probs = model.activate(images[batch_slice], 1.0);
        Array predictions = op::argmax(probs.w, -1);
        Array correct;
        if (labels.dtype() == DTYPE_INT32) {
            // labels are already integers.
            correct = labels.w[batch_slice];
        } else {
            // turn one-hots into labels
            correct = op::argmax((Array)labels.w[batch_slice], -1);
        }
        num_correct += op::sum(op::equals(predictions, correct));
        num_correct.eval();
    }
    return (Array)(num_correct.astype(DTYPE_DOUBLE) / num_images);
}

std::tuple<double, double> training_epoch(const MnistCnn& model,
                      std::shared_ptr<solver::AbstractSolver> solver,
                      Tensor images,
                      Tensor labels,
                      int batch_size) {
    int num_images = images.shape()[0].value();
    auto params = model.parameters();
    Array num_correct(0, DTYPE_DOUBLE);
    Array epoch_error(0, DTYPE_DOUBLE);
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        auto batch_slice = Slice(batch_start, std::min(batch_start + batch_size, num_images));
        Tensor batch_images = images[batch_slice];
        Tensor batch_labels = labels[batch_slice];
        batch_images.constant = true;
        Tensor probs = model.activate(batch_images, 0.5);
        Tensor error = tensor_ops::softmax_cross_entropy(probs, batch_labels);
        error.mean().grad();
        if (!FLAGS_use_jit_fusion) error.w.eval();
        epoch_error += error.w.sum();
        num_correct += op::sum(op::equals(op::argmax(probs.w, -1), batch_labels.w));
        graph::backward();
        op::control_dependencies(solver->step(params), {num_correct, epoch_error}).eval();
    }
    return std::make_tuple(
        (double)epoch_error / (double)num_images,
        (double)num_correct / (double)num_images
    );
}


std::vector<Tensor> load_dataset(const std::string& path) {
    auto train_x    = Tensor::load(utils::dir_join({path, "train_x.npy"}));
    auto train_y    = Tensor::load(utils::dir_join({path, "train_y.npy"}));

    auto validate_x = Tensor::load(utils::dir_join({path, "validate_x.npy"}));
    auto validate_y = Tensor::load(utils::dir_join({path, "validate_y.npy"}));

    auto test_x     = Tensor::load(utils::dir_join({path, "test_x.npy"}));
    auto test_y     = Tensor::load(utils::dir_join({path, "test_y.npy"}));

    train_x.constant = true;
    train_y.constant = true;

    validate_x.constant = true;
    validate_y.constant = true;

    test_x.constant = true;
    test_y.constant = true;

    return {train_x,    train_y,
            validate_x, validate_y,
            test_x,     test_y};
}


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
    const int batch_size = FLAGS_batch_size;
    WithCudnnPreference cudnn_pref(FLAGS_use_cudnn);
    op::jit::WithJITFusionPreference jit_pref(FLAGS_use_jit_fusion);
    std::cout << "Use CuDNN = " << (cudnn_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "Use JIT Fusion = " << (op::jit::jit_fusion_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "Use NCHW = " << (FLAGS_use_nchw ? "True" : "False") << "." << std::endl;
    std::cout << "loading dataset from " << FLAGS_path << "." << std::endl;
    auto ds = load_dataset(FLAGS_path);
    std::cout << "DONE." << std::endl;
    Tensor train_x    = ds[0], train_y    = ds[1],
           validate_x = ds[2], validate_y = ds[3],
           test_x     = ds[4], test_y     = ds[5];
    MnistCnn model(FLAGS_use_nchw);
    auto params = model.parameters();
    auto solver = solver::construct("sgd", params, 0.01);
    solver->clip_norm_ = 0.0;
    solver->clip_abs_  = 0.0;

    PerformanceReport report;
    double epoch_error, epoch_correct;

    long prev_number_of_computations = number_of_computations();
    long prev_number_of_allocations = memory::number_of_allocations();
    long prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
    long prev_actual_number_of_allocations = memory::bank::number_of_allocations();
    long prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();
    for (int i = 0; i < FLAGS_epochs; ++i) {
        auto epoch_start_time = std::chrono::system_clock::now();
        // report.start_capture();
        std::tie(epoch_error, epoch_correct) = training_epoch(model, solver, train_x, train_y, batch_size);
        // report.stop_capture();
        // report.print();
        std::chrono::duration<double> epoch_duration
                = (std::chrono::system_clock::now() - epoch_start_time);
        default_preferred_device.wait();
        // auto validate_acc = accuracy(model, validate_x, validate_y, batch_size);
        std::cout << epoch_duration.count()
                  << " " << number_of_computations() - prev_number_of_computations
                  << " " << memory::number_of_allocations() - prev_number_of_allocations
                  << " " << memory::number_of_bytes_allocated() - prev_number_of_bytes_allocated
                  << " " << memory::bank::number_of_allocations() - prev_actual_number_of_allocations
                  << " " << memory::bank::number_of_bytes_allocated() - prev_actual_number_of_bytes_allocated
                  << " " << num_expressions()
                  << std::endl;
        prev_number_of_computations = number_of_computations();
        prev_number_of_allocations = memory::number_of_allocations();
        prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
        prev_actual_number_of_allocations = memory::bank::number_of_allocations();
        prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();
        // std::cout << "Epoch " << i
        //           << ", train:      " << 100.0 * epoch_correct
        //           << " (nll " << epoch_error << ")"
        //           << ", validation: " << 100.0 * validate_acc << '%'
        //           << ", time:       " << epoch_duration.count() << "s" << std::endl;
    }
    // auto test_acc  = accuracy(model, test_x, test_y, batch_size);
    // std::cout << "Test accuracy: " << 100.0 * test_acc << '%' << std::endl;
}
