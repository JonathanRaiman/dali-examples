#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/utils/compiler.h>

#include "utils.h"

DEFINE_bool(use_cudnn, true, "Whether to use cudnn library for some GPU operations.");
DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_bool(recompile, false, "Whether to recompile ONNX loading.");
DEFINE_string(path, "", "Path to ONNX model.");
DEFINE_int32(batch_size, 256, "Batch size");
DEFINE_int32(epochs, 2, "Epochs");
DEFINE_int32(max_fusion_arguments, 3, "Max fusion arguments");

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

struct DaliSupportedDtype {
    std::string onnx_name;
    std::string dali_name;
    bool supported;
    DaliSupportedDtype(const std::string& onnx_name_) : onnx_name(onnx_name_), supported(false) {}
    DaliSupportedDtype(const std::string& onnx_name_, const std::string& dali_name_) : onnx_name(onnx_name_), dali_name(dali_name_), supported(true) {}
};

int main (int argc, char *argv[]) {
    typedef std::function<void(const std::vector<std::string>&, const std::vector<std::string>&, const std::string&, const std::string&, const std::string&)> node_extractor_t;
    typedef std::function<void(const Array&, const std::string&)> tensor_extractor_t;
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
        std::string include_proto_path = " -I/Users/jonathanraiman/Desktop/Coding/onnx_resources/proto/ -lprotobuf";
        std::vector<DaliSupportedDtype> mapping = {
            DaliSupportedDtype("UNDEFINED"), 
            DaliSupportedDtype("FLOAT", "DTYPE_FLOAT"), 
            DaliSupportedDtype("UINT8", "DTYPE_INT32"), 
            DaliSupportedDtype("INT8", "DTYPE_INT32"), 
            DaliSupportedDtype("UINT16", "DTYPE_INT32"), 
            DaliSupportedDtype("INT16", "DTYPE_INT32"), 
            DaliSupportedDtype("INT32", "DTYPE_INT32"), 
            DaliSupportedDtype("INT64", "DTYPE_INT32"), 
            DaliSupportedDtype("STRING"), 
            DaliSupportedDtype("BOOL"), 
            DaliSupportedDtype("FLOAT16", "DTYPE_FLOAT"), 
            DaliSupportedDtype("DOUBLE", "DTYPE_DOUBLE"), 
            DaliSupportedDtype("UINT32"), 
            DaliSupportedDtype("UINT64"), 
            DaliSupportedDtype("COMPLEX64"), 
            DaliSupportedDtype("COMPLEX128"), 
            DaliSupportedDtype("BFLOAT16")
        };
        std::stringstream ss;
        ss << ("#include <fstream>\n"
               "#include <string>\n"
               "#include <iostream>\n"
               "#include \"onnx.pb.h\"\n"
               "#include \"onnx.pb.cc\"\n"
               "#include \"dali/array/array.h\"\n"
               "#include \"dali/array/expression/buffer.h\"\n"
               "void run(") << get_function_arguments<const std::string&, node_extractor_t, tensor_extractor_t>() << (") {\n"
               "    onnx::ModelProto model;\n"
               "    std::fstream input(a, std::ios::in | std::ios::binary);\n"
               "    if (!model.ParseFromIstream(&input)) {\n"
               "      std::cerr << \"Failed to parse model.\" << std::endl;\n"
               "    } else {\n"
               "      std::cout << \"successfully parsed model!\" << std::endl;\n"
               "      auto& g = model.graph();\n"
               "      for (unsigned int i = 0, n = g.initializer_size(); i < n; i++) {\n"
               "          std::vector<Dim> dims;\n"
               "          auto& tensor = g.initializer(i);\n"
               "          for (unsigned int j = 0, ndims = tensor.dims_size(); j < ndims; j++) {\n"
               "              dims.emplace_back(tensor.dims(j));\n"
               "          }\n"
               "          DType dtype;\n"
               "          bool supported = false;\n"
               "          auto tensor_data_type = tensor.data_type();\n"
               "          switch (tensor_data_type) {\n");
        for (auto& k : mapping) {
            ss << "              case onnx::TensorProto::" << k.onnx_name << ":\n";
            if (k.supported) {
                ss << "                  dtype = " << k.dali_name << ";\n"
                      "                  supported = true;\n"
                      "                  break;\n";
            } else {
                ss << "                  std::cerr << \"Unsupported ONNX Dtype " << k.onnx_name << "\" << std::endl;\n"
                      "                  break;\n";
            }
        }
        ss << ("              default:\n"
               "                  break;\n"
               "          }\n"
               "          if (supported) {\n"
               "              Array arr(dims, dtype, Device::cpu());\n");

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

        for (auto& dtype_extractor : data_extraction_cases) {
            ss << "              if (tensor." << dtype_extractor << "_size() > 0) {\n";
            for (auto dtype_spec : dtype_specs) {
                auto cpp_dtype = dtype_to_cpp_name(std::get<0>(dtype_spec));
                ss << "                  if (dtype == " << std::get<1>(dtype_spec) << ") {\n"
                      "                      " << cpp_dtype << "* ptr = static_cast<" << cpp_dtype << "*>(op::static_as_buffer(arr.expression())->memory_->mutable_data(Device::cpu()));\n"
                      "                      for (unsigned int i = 0, n = tensor." << dtype_extractor << "_size(); i < n; i++) {\n"
                      "                          ptr[i] = tensor." << dtype_extractor << "(i);\n"
                      "                      }\n"
                      "                  }\n";
            }
            ss << "              }\n";
        }
        ss << ("              c(arr, tensor.name());\n"
               "          }\n"
               "      }\n"
               "      for (unsigned int i = 0, n = g.node_size(); i < n; i++) {\n"
               "          std::vector<std::string> inputs;\n"
               "          for (unsigned int j = 0, ninputs = g.node(i).input_size(); j < ninputs; j++) {\n"
               "              inputs.emplace_back(g.node(i).input(j));\n"
               "          }\n"
               "          std::vector<std::string> outputs;\n"
               "          for (unsigned int j = 0, noutputs = g.node(i).output_size(); j < noutputs; j++) {\n"
               "              outputs.emplace_back(g.node(i).output(j));\n"
               "          }\n"
               "          b(inputs, outputs, g.node(i).name(), g.node(i).op_type(), g.node(i).domain());\n"
               "      }\n"
               "    }\n"
               "}\n");
        compiler.compile<const std::string&, node_extractor_t, tensor_extractor_t>(hash, ss.str(), DEVICE_T_CPU, include_proto_path, false);
    }
    auto func = compiler.get_function<const std::string&, node_extractor_t, tensor_extractor_t>(hash);
    ASSERT2(!FLAGS_path.empty(), "Please provide a path to load an ONNX model.");
    int count = 0;
    func(FLAGS_path, [&count](const std::vector<std::string>& input, const std::vector<std::string>& output, const std::string& name, const std::string& op_type, const std::string& domain) {
        std::cout << "input: " << input << std::endl;
        std::cout << "output: " << output << std::endl;
        std::cout << "name: " << name << std::endl;
        std::cout << "op_type: " << op_type << std::endl;
        std::cout << "domain: " << domain << std::endl;
        std::cout << std::endl;
        count += 1;
    },
    [&count](const Array& array, const std::string& name) {
        if (array.is_stateless()) {
            std::cout << "stateless array" << std::endl;
        } else {
            std::cout << "shape: " << array.shape() << std::endl;
            std::cout << "dtype: " << array.dtype() << std::endl;
        }
        std::cout << "name: " << name << std::endl;
        std::cout << std::endl;
        count += 1;
    });
    std::cout << "Found " << count << " nodes!" << std::endl;
}
