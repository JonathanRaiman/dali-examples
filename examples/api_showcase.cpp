#include <iostream>
#include <vector>

#include "dali/core.h"
#include "dali/utils.h"

using std::vector;
using std::string;

// Test file for LSTM
int main () {
    LSTM lstm(30, 50,
        true   // do use Alex Graves' 2013 LSTM
               // where memory connects to gates
    );
    Tensor embedding = Tensor::uniform(-2.0, 2.0, {1000, 30});
    auto prev_state = lstm.initial_states();
    Tensor hidden, memory;
    std::tie(memory, hidden) = lstm.activate(embedding[vector<int>{0, 1, 10, 2, 1, 3}], prev_state);
    hidden.print();

    // load numpy matrix from file:
    auto name = "numpy_test.npy";
    std::cout << "loading a numpy matrix \"" << name << "\" from the disk" << std::endl;
    Tensor numpy_mat;
    if (utils::file_exists(name)) {
        numpy_mat = Tensor::load(name);
    } else {
        numpy_mat = Tensor({3, 3});
        for (int i = 0; i < 9; i++) numpy_mat.w(i) = i;
        Tensor::save(name, numpy_mat);
    }
    std::cout << "\"" << name << "\"=" << std::endl;
    // print it
    numpy_mat.print();
    // take softmax
    std::cout << "We now take a softmax of this matrix:" << std::endl;
    auto softmaxed = tensor_ops::softmax(numpy_mat, /*axis=*/0);
    softmaxed.print();
    int idx = 2;
    std::cout << "let us now compute the Kullback-Leibler divergence\n"
              << "between each column in this Softmaxed matrix and a\n"
              << "one-hot distribution peaking at index " << idx + 1 << "." << std::endl;

    // print softmax:
    Tensor idx_arr({}, DTYPE_INT32);
    idx_arr = idx_arr[Broadcast()];
    idx_arr.w = idx;

    auto divergence = tensor_ops::cross_entropy(softmaxed, idx_arr, /*axis=*/0);
    divergence.print();

    std::cout << "Press Enter to continue" << std::endl;
    getchar();

    Tensor A({3, 5});
    A += 1.2;
    A = A + Tensor::uniform(-0.5, 0.5, {3, 5});
    // build random matrix of double type with standard deviation 2:
    auto B = Tensor::uniform(-2.0, 2.0, {A.shape()[0], A.shape()[1]});
    auto C = Tensor::uniform(-2.0, 2.0, {A.shape()[1], 4       });

    A.print();
    B.print();

    auto A_times_B    = A * B;
    auto A_plus_B_sig = (A+B).sigmoid();
    auto A_dot_C_tanh = A.dot(C).tanh();
    auto A_plucked    = A[2];

    A_times_B.print();
    A_plus_B_sig.print();
    A_dot_C_tanh.print();
    A_plucked.print();

    // add some random singularity and use exponential
    // normalization:
    ELOG(A_plucked.shape());
    A_plucked.w[2] += 3.0;
    auto A_plucked_normed   = tensor_ops::softmax(A_plucked);
    auto A_plucked_normed_t = tensor_ops::softmax(A_plucked[Broadcast()], /*axis=*/0);
    A_plucked_normed.print();
    A_plucked_normed_t.print();

    // backpropagate to A and B
    auto params = lstm.parameters();

    StackedInputLayer superclassifier({20, 20, 10, 2}, 5);

    vector<Tensor> inputs;
    inputs.emplace_back(Tensor::uniform(-2.0, 2.0, {5, 20}));
    inputs.emplace_back(Tensor::uniform(-2.0, 2.0, {5, 20}));
    inputs.emplace_back(Tensor::uniform(-2.0, 2.0, {5, 10}));
    inputs.emplace_back(Tensor::uniform(-2.0, 2.0, {5, 2 }));

    auto out2 = superclassifier.activate(inputs);


    auto stacked = tensor_ops::hstack(inputs);
    auto stacked_a_b = tensor_ops::hstack({inputs[0], inputs[1]});

    stacked.print();
    stacked_a_b.print();

    out2.print();

    // Now vstacks:
    auto upper = Tensor::uniform(-2.0, 2.0, {4, 1});
    auto lower = Tensor::uniform(-2.0, 2.0, {4, 3});
    std::cout << "Stacking \"upper\": " << std::endl;
    upper.print();
    std::cout << "with \"lower\": " << std::endl;
    lower.print();
    std::cout << "using hstack(\"upper\", \"lower\") :" << std::endl;
    tensor_ops::hstack({upper, lower}).print();

    inputs[0] = tensor_ops::fill(inputs[0], 3);

    inputs[0].w[1] = 1;
    inputs[0].w[2] = 2;
    inputs[0].w[3] = 3;

    Tensor bob_indices({3,3}, DTYPE_INT32);

    bob_indices.w[0][0] = 1;
    bob_indices.w[1][0] = 2;
    bob_indices.w[2][0] = 3;

    auto bob = inputs[0][bob_indices[Slice(0,3)][0]];

    bob.print();

    auto dropped_bob = tensor_ops::dropout(bob, 0.2);

    dropped_bob.print();

    auto fast_dropped_bob = tensor_ops::fast_dropout(bob);

    fast_dropped_bob.print();

    std::cout << "Press enter to commence backpropagation." << std::endl;
    getchar();
    // TODO(szymon): fix the segfault in backprop.
    graph::backward();

    return 0;
}
