#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/tensor/op/spatial.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>

#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");


using std::vector;
using std::string;

const std::string MNIST_PATH = "/home/sidor/tmp/mnist/";



/* Simple convnet to learn MNIST digit recognition
 *
 * MNIST_PATH must point a directory containing
 * two files:
 *    - mnistX.npy - mnist images in form of
 *                   (70000, 784) tensor of floats
 *                   with values between 0 and 1
 *    - mnistY.npy - mnist labels corresponding
 *                   this those images represented
 *                   as an int array of shape
 *                   (70000,)
 */

Tensor shitty_softmax(Tensor x) {
    auto expted = x.exp();
    return expted / expted.sum(1)[Slice()][Broadcast()];
}

struct MnistCnn {
    ConvLayer conv1;
    ConvLayer conv2;
    Layer     fc1;
    Layer     fc2;

    MnistCnn() {
        conv1 = ConvLayer(32, 1,   5, 5);
        conv2 = ConvLayer(64, 32,  5, 5);
        fc1   = Layer(7 * 7 * 64, 1024);
        fc2   = Layer(1024,       10);
    }

    Tensor activate(Tensor images, float keep_prop) const {
        images = images.reshape({-1, 1, 28, 28});
        // shape (B, 1, 28, 28)

        Tensor out = conv1.activate(images).relu();
        out = tensor_ops::max_pool(out, 2, 2);
        // shape (B, 32, 14, 14)

        out = conv2.activate(out).relu();
        out = tensor_ops::max_pool(out, 2, 2);
        // shape (B, 64, 7, 7)

        out = out.reshape({out.shape()[0], 7 * 7 * 64});
        out = fc1.activate(out);
        // shape (B, 1024)

        out = tensor_ops::dropout(out, 1.0 - keep_prop);

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

    int num_images = images.shape()[0];

    auto num_correct = Array::zeros({}, DTYPE_INT32);
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        Slice batch_slice(batch_start, std::min(batch_start + batch_size, num_images));
        auto probs = model.activate(images[batch_slice], 1.0);
        Array predictions = op::astype(op::argmax(probs.w, -1), DTYPE_INT32);
        Array correct     = op::astype(op::argmax(labels.w[batch_slice], -1), DTYPE_INT32);
        num_correct += op::sum(op::equals(predictions, correct));
    }
    return (Array)(num_correct.astype(DTYPE_DOUBLE) / num_images);
}

double training_epoch(const MnistCnn& model,
                      std::shared_ptr<solver::AbstractSolver> solver,
                      Tensor images,
                      Tensor labels,
                      int batch_size) {
    int num_images = images.shape()[0];

    double epoch_error;

    auto params = model.parameters();

    for (int batch_start = 0;
            batch_start < images.shape()[0];
            batch_start+=batch_size) {
        auto batch_slice = Slice(batch_start, std::min(batch_start + batch_size, num_images));
        Tensor batch_images = images[batch_slice];
        Tensor batch_labels = labels[batch_slice];
        batch_images.constant = true;

        Tensor probs = model.activate(batch_images, 0.5);

        Tensor error = tensor_ops::cross_entropy(shitty_softmax(probs), batch_labels);
        error.grad();
        epoch_error += (double)(Array)error.w.sum();

        graph::backward();
        solver->step(params);
    }

    return epoch_error / (double)num_images;
}


std::vector<Tensor> load_dataset() {
    auto train_x    = Tensor::load(MNIST_PATH + "train_x.npy");
    auto train_y    = Tensor::load(MNIST_PATH + "train_y.npy");

    auto validate_x = Tensor::load(MNIST_PATH + "validate_x.npy");
    auto validate_y = Tensor::load(MNIST_PATH + "validate_y.npy");

    auto test_x     = Tensor::load(MNIST_PATH + "test_x.npy");
    auto test_y     = Tensor::load(MNIST_PATH + "test_y.npy");

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
        "\n"
        " @author Szymon Sidor\n"
        " @date July 4th 2016"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_device == -1) {
        memory::default_preferred_device = memory::Device::cpu();
    }
#ifdef DALI_USE_CUDA
    if (FLAGS_device >= 0) {
        memory::default_preferred_device = memory::Device::gpu(FLAGS_device);
    }
#endif

    utils::random::set_seed(123123);
    const int batch_size = 64;

    use_cudnn = FLAGS_use_cudnn;

    auto ds = load_dataset();
    Tensor train_x    = ds[0], train_y    = ds[1],
           validate_x = ds[2], validate_y = ds[3],
           test_x     = ds[4], test_y     = ds[5];

    MnistCnn model;

    auto params = model.parameters();
    auto solver = solver::construct("sgd", params, 0.01);
    solver->clip_norm = 0.0;
    solver->clip_abs  = 0.0;

    PerformanceReport report;

    for (int i = 0; i < 2; ++i) {
        auto epoch_start_time = std::chrono::system_clock::now();

        report.start_capture();
        auto epoch_error      = training_epoch(model, solver, train_x, train_y, batch_size);
        report.stop_capture();
        report.print();

        std::chrono::duration<double> epoch_duration
                = (std::chrono::system_clock::now() - epoch_start_time);
        auto validate_acc  = accuracy(model, validate_x, validate_y, batch_size);

        std::cout << "Epoch " << i
                  << ", train:    " << epoch_error
                  << ", valodate: " << 100.0 * validate_acc << '%'
                  << ", time:     " << epoch_duration.count() << "s" << std::endl;
    }

    auto test_acc  = accuracy(model, test_x, test_y, batch_size);
    std::cout << "Test accuracy: " << 100.0 * test_acc << '%' << std::endl;
}
