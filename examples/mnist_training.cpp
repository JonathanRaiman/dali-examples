#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/tensor/op/spatial.h>
#include <dali/tensor/layers/conv.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>
#include <dali/utils/make_message.h>

#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");
DEFINE_string(path, utils::dir_join({STR(DALI_EXAMPLES_DATA_DIR), "mnist"}), "Location of mnist data");
DEFINE_int32(batch_size, 256, "Batch size");

struct MnistCnn {
    ConvLayer conv1;
    ConvLayer conv2;
    Layer     fc1;
    Layer     fc2;

    MnistCnn() {
        conv1 = ConvLayer(32, 1, 5, 5);
        conv2 = ConvLayer(64, 32, 5, 5);
        fc1 = Layer(7 * 7 * 64, 1024);
        fc2 = Layer(1024, 10);
    }

    Tensor activate(Tensor images, float keep_prob) const {
        images = images.reshape({-1, 1, 28, 28});
        // shape (B, 1, 28, 28)
        Tensor out = conv1.activate(images).relu();
        out = tensor_ops::max_pool(out, 2, 2);
        // shape (B, 32, 14, 14)
        out = conv2.activate(out).relu();
        out = tensor_ops::max_pool(out, 2, 2);
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
                                   fc2.parameters()});
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
        error.w.eval();
        epoch_error += error.w.sum();
        epoch_error.eval();
        num_correct += op::sum(op::equals(op::argmax(probs.w, -1), batch_labels.w));
        num_correct.eval();
        graph::backward();
        solver->step(params);
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
    std::cout << "Use CuDNN = " << (cudnn_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "loading dataset from " << FLAGS_path << "." << std::endl;
    auto ds = load_dataset(FLAGS_path);
    std::cout << "DONE." << std::endl;
    Tensor train_x    = ds[0], train_y    = ds[1],
           validate_x = ds[2], validate_y = ds[3],
           test_x     = ds[4], test_y     = ds[5];
    MnistCnn model;
    auto params = model.parameters();
    auto solver = solver::construct("sgd", params, 0.01);
    solver->clip_norm_ = 0.0;
    solver->clip_abs_  = 0.0;

    PerformanceReport report;
    double epoch_error, epoch_correct;

    for (int i = 0; i < 10; ++i) {
        auto epoch_start_time = std::chrono::system_clock::now();

        report.start_capture();
        std::tie(epoch_error, epoch_correct) = training_epoch(model, solver, train_x, train_y, batch_size);
        report.stop_capture();
        report.print();

        std::chrono::duration<double> epoch_duration
                = (std::chrono::system_clock::now() - epoch_start_time);
        auto validate_acc = accuracy(model, validate_x, validate_y, batch_size);

        std::cout << "Epoch " << i
                  << ", train:      " << 100.0 * epoch_correct
                  << " (nll " << epoch_error << ")"
                  << ", validation: " << 100.0 * validate_acc << '%'
                  << ", time:       " << epoch_duration.count() << "s" << std::endl;
    }

    auto test_acc  = accuracy(model, test_x, test_y, batch_size);
    std::cout << "Test accuracy: " << 100.0 * test_acc << '%' << std::endl;
}
