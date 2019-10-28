#ifndef DALI_EXAMPLES_ONNX_ATTRIBUTE_H
#define DALI_EXAMPLES_ONNX_ATTRIBUTE_H

#include <vector>
#include <string>
#include "dali/array/dtype.h"

struct ONNXAttribute {
    enum AttributeType {
      UNDEFINED = 0,
      FLOAT = 1,
      INT = 2,
      STRING = 3,
      TENSOR = 4,
      GRAPH = 5,
      SPARSE_TENSOR = 11,

      FLOATS = 6,
      INTS = 7,
      STRINGS = 8,
      TENSORS = 9,
      GRAPHS = 10,
      SPARSE_TENSORS = 12
    };
    AttributeType type;
    float f;
    int i;
    std::string s;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
};

struct SymbolicDim {
    bool has_dim_value;
    int dim_value;
    std::string dim_param;
};

struct ONNXValueInfo {
    std::string name;
    bool is_tensor;
    DType dtype;
    std::vector<SymbolicDim> shape;
};

#endif