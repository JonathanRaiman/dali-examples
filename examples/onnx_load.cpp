#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/utils/compiler.h>
#include <dali/array/gemm/gemm_utils.h>

#include "onnx_attribute.h"
#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");
DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_bool(recompile, false, "Whether to recompile ONNX loading.");
DEFINE_string(path, "", "Path to ONNX model.");
DEFINE_int32(batch_size, 256, "Batch size");
DEFINE_int32(epochs, 2, "Epochs");
DEFINE_int32(max_fusion_arguments, 3, "Max fusion arguments");

typedef std::function<void(const std::vector<ONNXValueInfo>&)> value_info_extractor_t;
typedef std::unordered_map<std::string, ONNXAttribute> onnx_attr_t;
typedef std::function<void(const std::vector<std::string>&, const std::vector<std::string>&, const std::string&, const std::string&, const std::string&, const onnx_attr_t&)> node_extractor_t;
typedef std::function<void(const Array&, const std::string&)> tensor_extractor_t;

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

namespace {
    struct DaliSupportedDtype {
        std::string onnx_name;
        std::string dali_name;
        DType dali_dtype;
        bool supported;
        DaliSupportedDtype(const std::string& onnx_name_) : onnx_name(onnx_name_), supported(false) {}
        DaliSupportedDtype(const std::string& onnx_name_, const std::string& dali_name_, DType dtype) : onnx_name(onnx_name_), dali_name(dali_name_), supported(true), dali_dtype(dtype) {}
    };

    struct PoolMethod {
        std::string name;
        POOLING_T pooling_t;
        PoolMethod(const std::string& name_, POOLING_T pooling_t_) : name(name_), pooling_t(pooling_t_) {}
    };

    std::tuple<Array, PADDING_T> enforce_padding_policy(Array input, const onnx_attr_t& attributes) {
        PADDING_T padding = PADDING_T_VALID;
        if (attributes.find("pads") != attributes.end()) {
            auto& explicit_pads = DALI_MAPAT(attributes, "pads").ints;
            bool explicit_padding_info = false;
            for (auto v : explicit_pads) {
                if (v != 0) {
                    explicit_padding_info = true;
                }
            }
            if (explicit_padding_info) {
                shape_t paddings;
                for (auto& v : explicit_pads) {
                    paddings.emplace_back(v);
                }
                input = op::pad(input, paddings);
            }
        }
        if (attributes.find("auto_pad") != attributes.end()) {
            auto& pad_setting = DALI_MAPAT(attributes, "auto_pad").s;
            if (pad_setting == "NOTSET") {
                padding = PADDING_T_VALID;
            } else {
                if (pad_setting == "SAME_UPPER" || pad_setting == "SAME_LOWER") {
                    // TODO(jonathan): this is actually confounding two types of padding. We only support one kind.
                    // figure out which one.
                    padding = PADDING_T_SAME; 
                } else {
                    padding = PADDING_T_VALID;
                }
            }
        }
        return std::make_tuple(input, padding);
    }


    int get_int_with_default(const onnx_attr_t& attributes, const std::string& key, int default_value) {
        auto pos = attributes.find(key);
        return pos != attributes.end() ? pos->second.i : default_value;
    }

    std::vector<DaliSupportedDtype> onnx_dtype_mapping = {
        DaliSupportedDtype("UNDEFINED"), 
        DaliSupportedDtype("FLOAT", "DTYPE_FLOAT", DTYPE_FLOAT), 
        DaliSupportedDtype("UINT8", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("INT8", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("UINT16", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("INT16", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("INT32", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("INT64", "DTYPE_INT32", DTYPE_INT32), 
        DaliSupportedDtype("STRING"), 
        DaliSupportedDtype("BOOL"), 
        DaliSupportedDtype("FLOAT16", "DTYPE_FLOAT", DTYPE_FLOAT), 
        DaliSupportedDtype("DOUBLE", "DTYPE_DOUBLE", DTYPE_DOUBLE), 
        DaliSupportedDtype("UINT32"), 
        DaliSupportedDtype("UINT64"), 
        DaliSupportedDtype("COMPLEX64"), 
        DaliSupportedDtype("COMPLEX128"), 
        DaliSupportedDtype("BFLOAT16")
    };

    void convert_onnx_dtype_to_dali_dtype(int onnx_dtype, DType* dtype, bool* supported) {
        if (onnx_dtype < 0 || onnx_dtype >= onnx_dtype_mapping.size()) {
            *supported = false;
        }
        if (onnx_dtype_mapping[onnx_dtype].supported) {
          *dtype = onnx_dtype_mapping[onnx_dtype].dali_dtype;
          *supported = true;
        } else {
          *supported = false;
        }
    }
}


int main (int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "ONNX Loading\n"
        "------------\n");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_device == -1) {
        default_preferred_device = Device::cpu();
    }
    if (FLAGS_device >= 0) {
        ASSERT2(FLAGS_device < Device::num_gpus(), utils::make_message("Cannot run on GPU ", FLAGS_device, ", only found ", Device::num_gpus(), " gpus."));
        default_preferred_device = Device::gpu(FLAGS_device);
    }
    std::cout << "Running on " << default_preferred_device.description() << "." << std::endl;
    utils::random::set_seed(123123);

    auto& compiler = cpu_compiler;
    hash_t hash = 1132;
    if ((FLAGS_recompile || compiler.is_loaded(hash)) || !compiler.load(hash)) {
        // todo generalize path to protobuf file...
        std::string extra_compile_args = " -I/Users/jonathanraiman/Desktop/Coding/onnx_resources/proto/ -lprotobuf -I" + utils::dir_join({STR(DALI_EXAMPLES_SOURCE_DIR), "examples"});
        std::stringstream ss;
        std::vector<std::string> attr_types = {
            "UNDEFINED",
            "FLOAT",
            "INT",
            "STRING",
            "TENSOR",
            "GRAPH",
            "SPARSE_TENSOR",
            "FLOATS",
            "INTS",
            "STRINGS",
            "TENSORS",
            "GRAPHS",
            "SPARSE_TENSORS"
        };
        ss << ("#include <fstream>\n"
               "#include <string>\n"
               "#include <iostream>\n"
               "#include <google/protobuf/io/zero_copy_stream_impl.h>\n"
               "#include \"onnx.pb.h\"\n"
               "#include \"onnx.pb.cc\"\n"
               "#include \"dali/array/array.h\"\n"
               "#include \"onnx_attribute.h\"\n"
               "\n"
               "ONNXAttribute::AttributeType convert_onnx_attribute_type_to_dali_attribute_type(onnx::AttributeProto::AttributeType type) {\n"
               "    switch (type) {\n");
        for (auto& attr_type : attr_types) {
          ss << "       case onnx::AttributeProto::" << attr_type << ":\n"
                "           return ONNXAttribute::AttributeType::" << attr_type << ";\n";
        }
        ss << ("        default:\n"
               "             return ONNXAttribute::AttributeType::UNDEFINED;\n"
               "    }\n"
               "}\n");
        ss << "onnx::TensorProto::DataType int_to_onnx_dtype(int value) {\n"
              "    switch (value) {\n";
        for (int i = 0; i < onnx_dtype_mapping.size(); i++) {
            ss << "        case " << i << ":\n"
                  "            return onnx::TensorProto::" << onnx_dtype_mapping[i].onnx_name << ";\n";
        }
        ss << "        default:\n"
              "            return onnx::TensorProto::UNDEFINED;\n"
              "    }\n"
              "}\n";
        ss << ("void convert_onnx_dtype_to_dali_dtype(const onnx::TensorProto::DataType& onnx_dtype, DType* dtype, bool* supported) {\n"
               "    *supported = false;\n"
               "    switch (onnx_dtype) {\n");
        for (auto& k : onnx_dtype_mapping) {
            ss << "        case onnx::TensorProto::" << k.onnx_name << ":\n";
            if (k.supported) {
                ss << "            *dtype = " << k.dali_name << ";\n"
                      "            *supported = true;\n"
                      "            break;\n";
            } else {
                ss << "            std::cerr << \"Unsupported ONNX Dtype " << k.onnx_name << "\" << std::endl;\n"
                      "            break;\n";
            }
        }
        ss << ("        default:\n"
               "            break;\n"
               "    }\n"
               "}\n"
               "\n"
               "ONNXAttribute parse_attribute(const onnx::AttributeProto& attr) {\n"
               "    ONNXAttribute dali_attribute;\n"
               "    dali_attribute.type = convert_onnx_attribute_type_to_dali_attribute_type(attr.type());\n"
               "    switch (dali_attribute.type) {\n"
               "        case ONNXAttribute::AttributeType::INTS:\n"
               "            for (unsigned int i = 0, n = attr.ints_size(); i < n; i++) {\n"
               "                dali_attribute.ints.emplace_back(attr.ints(i));\n"
               "            }\n"
               "            break;\n"
               "        case ONNXAttribute::AttributeType::FLOATS:\n"
               "            for (unsigned int i = 0, n = attr.floats_size(); i < n; i++) {\n"
               "                dali_attribute.floats.emplace_back(attr.floats(i));\n"
               "            }\n"
               "            break;\n"
               "        case ONNXAttribute::AttributeType::STRINGS:\n"
               "            for (unsigned int i = 0, n = attr.strings_size(); i < n; i++) {\n"
               "                dali_attribute.strings.emplace_back(attr.strings(i));\n"
               "            }\n"
               "            break;\n"
               "        case ONNXAttribute::AttributeType::STRING:\n"
               "            dali_attribute.s = attr.s();\n"
               "            break;\n"
               "        case ONNXAttribute::AttributeType::INT:\n"
               "            dali_attribute.i = attr.i();\n"
               "            break;\n"
               "        case ONNXAttribute::AttributeType::FLOAT:\n"
               "            dali_attribute.f = attr.f();\n"
               "            break;\n"
               "        default:"
               "            break;\n"
               "    }\n"
               "    return dali_attribute;\n"
               "}\n"
               "\n"
               "void parse_onnx_value_info(ONNXValueInfo& out, const onnx::ValueInfoProto& value_info) {\n"
               "    out.name = value_info.name();\n"
               "    if (!value_info.type().has_tensor_type()) {\n"
               "        out.is_tensor = false;\n"
               "        return;\n"
               "    }\n"
               "    out.is_tensor = true;\n"
               "    bool supported;\n"
               "    convert_onnx_dtype_to_dali_dtype(int_to_onnx_dtype(value_info.type().tensor_type().elem_type()), &out.dtype, &supported);\n"
               "    auto& shape = value_info.type().tensor_type().shape();\n"
               "    out.shape.resize(shape.dim_size());\n"
               "    for (unsigned int i = 0, n = shape.dim_size(); i < n; i++) {\n"
               "        out.shape[i].has_dim_value = shape.dim(i).has_dim_value();\n"
               "        if (shape.dim(i).has_dim_value()) {\n"
               "            out.shape[i].dim_value = shape.dim(i).dim_value();\n"
               "        } else {\n"
               "            out.shape[i].dim_param = shape.dim(i).dim_param();\n"
               "        }\n"
               "    }\n"
               "}\n"
               "\n"
               "void run(") << get_function_arguments<int, const std::string&, value_info_extractor_t, value_info_extractor_t, node_extractor_t, tensor_extractor_t, bool*>(
                    {"total_bytes_limit", "path", "value_info_input_extractor", "value_info_output_extractor", "node_extractor", "tensor_extractor", "error_ptr"}) << (") {\n"
               "    onnx::ModelProto model;\n"
               "    std::fstream input(path, std::ios::in | std::ios::binary);\n"
               "    google::protobuf::io::IstreamInputStream rawInput(&input);\n"
               "    google::protobuf::io::CodedInputStream coded_stream(&rawInput);\n"
               "    coded_stream.SetTotalBytesLimit(total_bytes_limit, total_bytes_limit / 2);\n"
               "    if (!model.ParseFromCodedStream(&coded_stream)) {\n"
               "      *error_ptr = true;\n"
               "      return;\n"
               "    }\n"
               "    *error_ptr = false;\n"
               "    auto& g = model.graph();\n"
               "    for (unsigned int i = 0, n = g.initializer_size(); i < n; i++) {\n"
               "        std::vector<Dim> dims;\n"
               "        auto& tensor = g.initializer(i);\n"
               "        for (unsigned int j = 0, ndims = tensor.dims_size(); j < ndims; j++) {\n"
               "            dims.emplace_back(tensor.dims(j));\n"
               "        }\n"
               "        DType dtype;\n"
               "        bool supported = false;\n"
               "        convert_onnx_dtype_to_dali_dtype(int_to_onnx_dtype(tensor.data_type()), &dtype, &supported);\n"
               "        if (supported) {\n"
               "            Array arr(dims, dtype, Device::cpu());\n");

        std::vector<std::string> data_extraction_cases = {
            "float_data",
            "int32_data",
            "int64_data",
            "double_data",
            "uint64_data"
        };

        std::vector<std::tuple<DType, std::string>> dtype_specs = {
          {DTYPE_INT32, "DTYPE_INT32"},
          {DTYPE_FLOAT, "DTYPE_FLOAT"},
          {DTYPE_DOUBLE, "DTYPE_DOUBLE"}
        };

        // TODO(jonathan): there are cases where the data is stored externally
        for (auto& dtype_extractor : data_extraction_cases) {
            ss << "            if (tensor." << dtype_extractor << "_size() > 0) {\n";
            for (auto dtype_spec : dtype_specs) {
                auto cpp_dtype = dtype_to_cpp_name(std::get<0>(dtype_spec));
                ss << "                if (dtype == " << std::get<1>(dtype_spec) << ") {\n"
                      "                    " << cpp_dtype << "* ptr = static_cast<" << cpp_dtype << "*>(arr.memory()->mutable_data(Device::cpu()));\n"
                      "                    for (unsigned int i = 0, n = tensor." << dtype_extractor << "_size(); i < n; i++) {\n"
                      "                        ptr[i] = tensor." << dtype_extractor << "(i);\n"
                      "                    }\n"
                      "                }\n";
            }
            ss << "              }\n";
        }
        ss << ("            tensor_extractor(arr, tensor.name());\n"
               "        }\n"
               "    }\n"
               "    {\n"
               "        std::vector<ONNXValueInfo> args(g.input_size());\n"
               "        for (unsigned int i = 0, n = g.input_size(); i < n; i++) {\n"
               "            parse_onnx_value_info(args[i], g.input(i));\n"
               "        }\n"
               "        value_info_input_extractor(args);\n"
               "    }\n"
               "    for (unsigned int i = 0, n = g.node_size(); i < n; i++) {\n"
               "        auto& node = g.node(i);\n"
               "        std::vector<std::string> inputs;\n"
               "        for (unsigned int j = 0, ninputs = node.input_size(); j < ninputs; j++) {\n"
               "            inputs.emplace_back(node.input(j));\n"
               "        }\n"
               "        std::vector<std::string> outputs;\n"
               "        for (unsigned int j = 0, noutputs = node.output_size(); j < noutputs; j++) {\n"
               "            outputs.emplace_back(node.output(j));\n"
               "        }\n"
               "        std::unordered_map<std::string, ONNXAttribute> attributes;\n"
               "        for (unsigned int j = 0, nattributes = node.attribute_size(); j < nattributes; j++) {\n"
               "            attributes.emplace(node.attribute(j).name(), parse_attribute(node.attribute(j)));\n"
               "        }\n"
               "        node_extractor(inputs, outputs, node.name(), node.op_type(), node.domain(), attributes);\n"
               "    }\n"
               "    {\n"
               "        std::vector<ONNXValueInfo> args(g.output_size());\n"
               "        for (unsigned int i = 0, n = g.output_size(); i < n; i++) {\n"
               "            parse_onnx_value_info(args[i], g.output(i));\n"
               "        }\n"
               "        value_info_output_extractor(args);\n"
               "    }\n"
               "}\n");
        compiler.compile<int, const std::string&, value_info_extractor_t, value_info_extractor_t, node_extractor_t, tensor_extractor_t, bool*>(hash, ss.str(), DEVICE_T_CPU, extra_compile_args, false);
    }
    auto func = compiler.get_function<int, const std::string&, value_info_extractor_t, value_info_extractor_t, node_extractor_t, tensor_extractor_t, bool*>(hash);
    ASSERT2(!FLAGS_path.empty(), "Please provide a path to load an ONNX model.");
    int count = 0;

    std::unordered_map<std::string, Array> name2array;
    std::unordered_map<std::string, std::function<std::vector<Array>(const std::vector<Array>&, const onnx_attr_t&)>> name2op;

    name2op.emplace("MatMul", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 argument for MatMul.");
        return {op::dot(args[0], args[1])};
    });
    name2op.emplace("Gemm", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2 || args.size() == 3, "Expected 2 or arguments for Gemm.");
        // Scalar multiplier for the product of input tensors A * B.
        ASSERT2(args[0].ndim() == 2 && args[1].ndim() == 2, "Input arguments to Gemm must be of dimension 2.");
        auto a = args[0];
        auto b = args[1];
        if (attributes.find("transA") != attributes.end() && DALI_MAPAT(attributes, "transA").i == 1) {
          a = a.transpose();
        }
        if (attributes.find("transB") != attributes.end() && DALI_MAPAT(attributes, "transB").i == 1) {
          b = b.transpose();
        }
        Array out = op::dot(a, b);
        if (attributes.find("alpha") != attributes.end() && DALI_MAPAT(attributes, "alpha").f != 1.0) {
          out = make_expression<op::MatMul>(out.expression()->arguments()[0], out.expression()->arguments()[1],
                                            Array(DALI_MAPAT(attributes, "alpha").f, out.dtype()).expression(), out.shape());
        }
        if (args.size() == 3) {
            // Scalar multiplier for input tensor C.
            if (attributes.find("beta") != attributes.end() && DALI_MAPAT(attributes, "beta").f != 1.0) {
              out = out + args[2] * DALI_MAPAT(attributes, "beta").f;
            } else {
              out = out + args[2];
            }
        }   
        return {out};
    });
    name2op.emplace("Relu", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Relu.");
        return {op::relu(args[0])};
    }); 
    name2op.emplace("Sinh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Sinh.");
        return {op::sinh(args[0])};
    });
    name2op.emplace("Cosh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Cosh.");
        return {op::cosh(args[0])};
    });
    name2op.emplace("Tanh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Tanh.");
        return {op::tanh(args[0])};
    });
    name2op.emplace("Asinh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Asinh.");
        return {op::asinh(args[0])};
    });
    name2op.emplace("Acosh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Acosh.");
        return {op::acosh(args[0])};
    });
    name2op.emplace("Atanh", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Atanh.");
        return {op::atanh(args[0])};
    });
    name2op.emplace("Neg", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Neg.");
        return {op::negate(args[0])};
    });
    name2op.emplace("Log", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Log.");
        return {op::log(args[0])};
    });
    name2op.emplace("Sqrt", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Sqrt.");
        return {op::sqrt(args[0])};
    });
    name2op.emplace("Reciprocal", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Reciprocal.");
        return {op::eltinv(args[0])};
    });
    name2op.emplace("Sign", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Sign.");
        return {op::sign(args[0])};
    });
    name2op.emplace("Erf", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Erf.");
        return {op::erf(args[0])};
    });
    name2op.emplace("Sin", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Sin.");
        return {op::sin(args[0])};
    });
    name2op.emplace("Cos", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Cos.");
        return {op::cos(args[0])};
    });
    name2op.emplace("Tan", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Tan.");
        return {op::tan(args[0])};
    });
    name2op.emplace("Asin", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Asin.");
        return {op::asin(args[0])};
    });
    name2op.emplace("Acos", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Acos.");
        return {op::acos(args[0])};
    });
    name2op.emplace("Atan", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Atan.");
        return {op::atan(args[0])};
    });
    name2op.emplace("Exp", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Exp.");
        return {op::exp(args[0])};
    });
    name2op.emplace("Sigmoid", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Sigmoid.");
        return {op::sigmoid(args[0])};
    });
    name2op.emplace("Softplus", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Softplus.");
        return {op::softplus(args[0])};
    });
    name2op.emplace("Softsign", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Softsign.");
        return {op::softsign(args[0])};
    });
    name2op.emplace("Abs", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Abs.");
        return {op::abs(args[0])};
    });
    name2op.emplace("Add", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Add.");
        return {args[0] + args[1]};
    });
    name2op.emplace("Sum", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        return {op::add(args)};
    });
    name2op.emplace("Sub", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Sub.");
        return {args[0] - args[1]};
    });
    name2op.emplace("Div", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Div.");
        return {args[0] / args[1]};
    });
    name2op.emplace("Mul", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Mul.");
        return {args[0] * args[1]};
    });
    name2op.emplace("Pow", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Pow.");
        return {op::pow(args[0], args[1])};
    });
    
    name2op.emplace("Reshape", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Reshape.");
        ASSERT2(args[1].is_buffer(), "Expected argument 2 of Reshape to be a buffer");
        ASSERT2(args[1].dtype() == DTYPE_INT32, "Expected argument 2 of Reshape to be integers.");
        ASSERT2(args[1].ndim() == 1, "Expected argument 2 of Reshape to be 1-dimensional.");
        std::vector<Dim> dims;
        args[1].print();
        const int* shape_ptr = static_cast<int*>(args[1].memory()->readonly_data(Device::cpu()));
        for (int i = 0; i < args[1].shape()[0]; i++) {
          if (shape_ptr[i] == 0) {
              // From ONNX spec:
              // "A dimension could also be 0, in which case the actual
              // dimension value is unchanged (i.e. taken from the input
              // tensor)"
              if (args[0].shape().size() > i) {
                  // ASSERT2(args[0].shape().size() > i,
                  //         utils::make_message("Attempting to copy over input dimension at axis ",
                  //                             i, " but input has shape ", args[0].shape(), "."));
                  dims.emplace_back(args[0].shape()[i]);
              }
          } else {
              dims.emplace_back(shape_ptr[i]);
          }
        }
        return {args[0].reshape(dims)};
    });
    name2op.emplace("Conv", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 2, "Expected 2 arguments for Conv.");
        ASSERT2(DALI_MAPAT(attributes, "group").i == 1, "Currently only group=1 attribute supported for conv.");
        for (auto& v : DALI_MAPAT(attributes, "dilations").ints) {
            ASSERT2(v == 1, "Currently only dilation=1 attribute supported for conv.");
        }
        auto& strides = DALI_MAPAT(attributes, "strides").ints;
        ASSERT2(strides.size() == 2, utils::make_message("Expected strides attribute to be of length 2, but got ", strides, " instead."));

        Array input = args[0];
        PADDING_T padding;
        std::tie(input, padding) = enforce_padding_policy(input, attributes);

        return {op::conv2d(input, args[1], strides[0], strides[1], padding, "NCHW")};
    });

    // TODO: implement
    // And
    // Argmax
    // Argmin
    // BatchNormalization
    // BitShift
    // Ceil,
    // Clip,
    // Compress,
    // Concat,
    // ConcatFromSequence,
    // Constant,
    // ConstantOfShape,
    // ConvTranspose,
    // CumSum,
    // DepthToSpace,
    // Det
    // Dropout
    // Elu
    // Equal
    // Expand
    // EyeLike
    // Floor
    // GRU
    // Gather
    // GatherElements
    // GatherND
    // GlobalLpPool
    // Greater
    // HardSigmoid
    // Hardmax
    // If
    // InstanceNormalization
    // Isinf
    // IsNan
    // LRN
    // LSTM
    // LeakyRelu
    // Less
    // LogSoftmax
    // Loop
    // LpNormalization
    // LpPool
    // MaxRoiPool
    // MaxUnpool
    // Min (https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min)
    // Mod
    // Multinomial
    // NonMaxSuppression
    // Nonzero
    // Not
    // OneHot
    // Or
    // PRelu
    // Pad
    // RNN,
    // RandomNormal
    // RandomNormalLike
    // RandomUniform
    // RandomUniformLike
    // Range,
    // ReduceL1,
    // ReduceLogSum,
    // ReduceLogSumExp,
    // ReduceSumSquare,
    // Resize,
    // ReverseSequence,
    // RoiAlign,
    // Round,
    // Scan,
    // Scatter,
    // ScatterElements,
    // ScatterND,
    // Selu,
    // SequenceAt,
    // SequenceConstruct,
    // SequenceEmpty,
    // SequenceErase,
    // SequenceInsert,
    // SequenceLength,
    // Shape,
    // Shrink,
    // Size,
    // Slice (https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice)
    // Softmax (https://github.com/onnx/onnx/blob/master/docs/Operators.md#softmax),
    // SpaceToDepth,
    // Split,
    // SplitToSequence,
    // Squeeze,
    // StringNormalizer,
    // TfIdfVectorizer,
    // ThresholdedRelu,
    // Tile,
    // TopK,
    // Unique,
    // Unsqueeze,
    // Where,
    // XOR

    // Reshape is messed up (reshape with dimensions 0 doesn't make much sense)

    name2op.emplace("BatchNormalization", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 5, "Expected 5 arguments for BatchNormalization.");
        // TODO(jonathan): implement
        return {args[0]};
    });
    name2op.emplace("Identity", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Identity.");
        // TODO(jonathan): implement
        return {args[0]};
    });

    name2op.emplace("Cast", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Cast.");
        auto& onnx_to_dtype = DALI_MAPAT(attributes, "to").i;
        DType dtype;
        bool supported;
        convert_onnx_dtype_to_dali_dtype(onnx_to_dtype, &dtype, &supported);
        if (onnx_to_dtype > 0 && onnx_to_dtype < onnx_dtype_mapping.size()) {
            ASSERT2(supported, utils::make_message("Could not build Cast, ONNX dtype ", onnx_to_dtype, " (", onnx_dtype_mapping[onnx_to_dtype].onnx_name, ") not yet supported."));
        } else {
            ASSERT2(supported, utils::make_message("Could not build Cast, ONNX dtype ", onnx_to_dtype, " not yet supported."));
        }
        return {args[0].astype(dtype)};
    });
    name2op.emplace("Transpose", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Transpose.");
        if (attributes.find("perm") != attributes.end() && DALI_MAPAT(attributes, "perm").ints.size() > 0) {
            return {args[0].transpose(DALI_MAPAT(attributes, "perm").ints)};
        } else {
            return {args[0].transpose()};
        }
    });
    name2op.emplace("GlobalAveragePool", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for GlobalAveragePool.");
        std::vector<int> axes;
        for (int i = 2; i < args[0].ndim(); i++) {
          axes.emplace_back(i);
        }
        return {op::mean(args[0], axes, /*keepdims=*/true)};
    });
    name2op.emplace("GlobalMaxPool", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for GlobalMaxPool.");
        std::vector<int> axes;
        for (int i = 2; i < args[0].ndim(); i++) {
          axes.emplace_back(i);
        }
        return {op::max(args[0], axes, /*keepdims=*/true)};
    });
    name2op.emplace("GlobalLpPool", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for GlobalLpPool.");
        auto& p = DALI_MAPAT(attributes, "p").i;
        ASSERT2(p == 2, utils::make_message("Only L2 norm is supported but got p = ", p, "."));
        std::vector<int> axes;
        for (int i = 2; i < args[0].ndim(); i++) {
          axes.emplace_back(i);
        }
        return {op::L2_norm(args[0], axes, /*keepdims=*/true)};
    });
    name2op.emplace("ReduceL2", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceL2.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::L2_norm(args[0], axes, /*keepdims=*/keepdims)};
    });
    name2op.emplace("ReduceMax", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceMax.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::max(args[0], axes, /*keepdims=*/keepdims)};
    });
    name2op.emplace("ReduceMin", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceMin.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::min(args[0], axes, /*keepdims=*/keepdims)};
    });
    name2op.emplace("ReduceMean", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceMean.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::mean(args[0], axes, /*keepdims=*/keepdims)};
    });
    name2op.emplace("ReduceProd", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceProd.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::prod(args[0], axes, /*keepdims=*/keepdims)};
    });
    name2op.emplace("ReduceSum", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for ReduceSum.");
        auto& axes = DALI_MAPAT(attributes, "axes").ints;
        bool keepdims = get_int_with_default(attributes, "keepdims", true) == 1;
        return {op::sum(args[0], axes, /*keepdims=*/keepdims)};
    });

    name2op.emplace("Flatten", [](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
        ASSERT2(args.size() == 1, "Expected 1 argument for Flatten.");
        int axis = 1;
        if (attributes.find("axis") != attributes.end()) {
          axis = DALI_MAPAT(attributes, "axis").i;
        }
        shape_t output_shape;
        if (axis == 0) {
          output_shape.emplace_back(Dim::one());
        } else {
          auto leading_dim = args[0].shape()[0];
          for (int i = 1; i < axis; i++) {
            leading_dim *= args[0].shape()[i];
          }
          output_shape.emplace_back(leading_dim);
        }
        if (axis == args[0].ndim()) {
          output_shape.emplace_back(Dim::one());
        } else {
          auto last_dim = args[0].shape()[axis];
          for (int i = axis + 1; i < args[0].ndim(); i++) {
            last_dim *= args[0].shape()[i];
          }
          output_shape.emplace_back(last_dim);
        }
        return {args[0].reshape(output_shape)};
    });

    std::vector<PoolMethod> pools = {
      PoolMethod("MaxPool", POOLING_T_MAX),
      PoolMethod("AveragePool", POOLING_T_AVG)
    };
    for (auto& pool_data : pools) {
      name2op.emplace(pool_data.name, [pool_data](const std::vector<Array>& args, const onnx_attr_t& attributes) -> std::vector<Array> {
          ASSERT2(args.size() == 1, utils::make_message("Expected 1 argument for ", pool_data.name, "."));
          auto& kernel_shape = DALI_MAPAT(attributes, "kernel_shape").ints;
          ASSERT2(kernel_shape.size() == 2, utils::make_message("Expected kernel_shape attribute to be of length 2, but got ", kernel_shape, " instead."));
          auto& strides = DALI_MAPAT(attributes, "strides").ints;
          ASSERT2(strides.size() == 2, utils::make_message("Expected strides attribute to be of length 2, but got ", strides, " instead."));

          Array input = args[0];
          PADDING_T padding;
          std::tie(input, padding) = enforce_padding_policy(input, attributes);

          return {op::pool2d(input,
                             kernel_shape[0],
                             kernel_shape[1],
                             strides[0],
                             strides[1],
                             pool_data.pooling_t,
                             padding,
                             "NCHW")};
      });
    }


    int missing_array = 0;
    std::unordered_set<std::string> missing_ops;
    std::vector<Array> placeholders;
    std::vector<Array> outputs;
    bool error = true;
    // 1 gigabyte
    int total_bytes_limit = 1073741824;
    func(
        total_bytes_limit,
        FLAGS_path,
        [&name2array, &placeholders](const std::vector<ONNXValueInfo>& input_args) {
            // input to graph
            // place those into the graph??
            for (auto& i : input_args) {
                if (name2array.find(i.name) == name2array.end()) {
                    if (i.is_tensor) {
                        std::cout << "Found placeholder input: " << i.name << std::endl;
                        std::cout << "                  shape_size " << i.shape.size() << std::endl;
                        std::cout << "                  dtype " << i.dtype << std::endl;
                        std::vector<Dim> shape;
                        std::vector<std::string> dims;
  
                        for (auto& v : i.shape) {
                          if (v.has_dim_value) {
                            dims.emplace_back(utils::make_message(v.dim_value));
                            shape.emplace_back(v.dim_value);
                          } else {
                            dims.emplace_back(v.dim_param);
                            // we don't know the shape, so let's put 1 for now...
                            shape.emplace_back(1);
                          }
                        }
                        std::cout << "                  " << dims << std::endl;
                        std::cout << std::endl;
                        Array pholder(shape, i.dtype);
                        placeholders.emplace_back(pholder);
                        name2array.emplace(i.name, pholder);
                    }
                }
            }
        },
        [&name2array, &outputs](const std::vector<ONNXValueInfo>& output_args) {
            for (auto& i : output_args) {
                if (i.is_tensor && name2array.find(i.name) != name2array.end()) {
                    outputs.emplace_back(DALI_MAPAT(name2array, i.name));
                }
            }
        },
        [&count, &name2array, &name2op, &missing_array, &missing_ops](const std::vector<std::string>& onnx_inputs,
                                                                      const std::vector<std::string>& onnx_outputs,
                                                                      const std::string& name,
                                                                      const std::string& op_type,
                                                                      const std::string& domain,
                                                                      const onnx_attr_t& attributes) {
            std::cout << "input: " << onnx_inputs << std::endl;
            std::cout << "output: " << onnx_outputs << std::endl;
            std::cout << "name: " << name << std::endl;
            std::cout << "op_type: " << op_type << std::endl;
            for (auto& attr : attributes) {
                std::cout << "    attribute " << attr.first << std::endl;
                switch (attr.second.type) {
                  case ONNXAttribute::AttributeType::INTS:
                    std::cout << "              " << attr.second.ints << std::endl;
                    break;
                  case ONNXAttribute::AttributeType::FLOATS:
                    std::cout << "              " << attr.second.floats << std::endl;
                    break;
                  case ONNXAttribute::AttributeType::STRINGS:
                    std::cout << "              " << attr.second.strings << std::endl;
                    break;
                  case ONNXAttribute::AttributeType::STRING:
                    std::cout << "              " << attr.second.s << std::endl;
                    break;
                  case ONNXAttribute::AttributeType::INT:
                    std::cout << "              " << attr.second.i << std::endl;
                    break;
                  case ONNXAttribute::AttributeType::FLOAT:
                    std::cout << "              " << attr.second.f << std::endl;
                    break;
                  default:
                    std::cout << "              unsupported type" << std::endl;
                    break;
                }
            }
            std::cout << std::endl;
            count += 1;
            std::vector<Array> dali_inputs;
            for (auto& a : onnx_inputs) {
                if (name2array.find(a) == name2array.end()) {
                    std::cout << "Missing " << a << "!!" << std::endl;
                    missing_array += 1;
                } else {
                    dali_inputs.emplace_back(DALI_MAPAT(name2array, a));
                }
            }
            if (dali_inputs.size() == onnx_inputs.size()) {
                if (name2op.find(op_type) == name2op.end()) {
                    std::cout << "Missing operation " << op_type << "!!" << std::endl;
                    missing_ops.emplace(op_type);
                } else {
                    auto dali_outputs = DALI_MAPAT(name2op, op_type)(dali_inputs, attributes);
                    ASSERT2(dali_outputs.size() == onnx_outputs.size(), utils::make_message(
                            "Not the same number of outputs given by Dali (", dali_outputs.size(), ") as ONNX (", onnx_outputs.size(), ")."));
                    for (int i = 0; i < dali_outputs.size(); i++) {
                        name2array.emplace(onnx_outputs[i], dali_outputs[i]);
                    }
                }
            }
        },
        [&count, &name2array](const Array& array, const std::string& name) {
            // if (array.is_stateless()) {
            //     std::cout << "stateless array" << std::endl;
            // } else {
            //     std::cout << "shape: " << array.shape() << std::endl;
            //     std::cout << "dtype: " << array.dtype() << std::endl;
            // }
            // std::cout << "name: " << name << std::endl;
            // std::cout << std::endl;
            count += 1;
            name2array.emplace(name, array);
        }, &error);
    if (error) {
        std::cerr << "Failed to parse model!" << std::endl;
    } else {
        std::cout << "successfully parsed model!" << std::endl;
        std::cout << "Found " << count << " nodes!" << std::endl;
        if (missing_ops.empty() && missing_array == 0) {
            for (auto& out : outputs) {
                out.print();
            }
        } else {
            std::cout << "Missing ops " << std::vector<std::string>(missing_ops.begin(), missing_ops.end()) << std::endl;
            std::cout << "Missing " << missing_array << " arrays" << std::endl;
        }
    }
}
