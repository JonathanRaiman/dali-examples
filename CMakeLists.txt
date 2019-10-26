cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali-examples")

# additional cmake find files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(DALI_EXAMPLES_DATA_DIR "${PROJECT_SOURCE_DIR}/data")


if (DEFINED ENV{DALI_HOME} AND NOT "$ENV{DALI_HOME}" STREQUAL "")
    set(Dali_DIR $ENV{DALI_HOME}/cmake)
endif()

# find Dali.
find_package(Dali REQUIRED)
include_directories(${DALI_AND_DEPS_INCLUDE_DIRS})

# find gflags
find_package(Gflags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

# ensure we are using C++11
# TODO(jonathan): is this flag required for apple?
# -stdlib=libc++
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++14 -O3 -g -fPIC" )

include_directories(${PROJECT_SOURCE_DIR})

# TODO(szymon): reenable once models are ported
# file(GLOB_RECURSE UTILS_FILES "${PROJECT_SOURCE_DIR}/models/*.cpp")

list(APPEND       UTILS_FILES "examples/utils.h")
add_library(utils examples/utils.cpp ${UTILS_FILES})
target_link_libraries(utils ${CMAKE_DL_LIBS})

# Compile all the examples
set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(EXAMPLES_BUILD_DIR ${PROJECT_SOURCE_DIR}/build/examples)
set(EXAMPLE_PROGRAMS # api_showcase
                     # arithmetic
                     # beam_tree_training
                     # bidirectional_sentiment
                     # character_prediction
                     # language_model
                     # lstm_sentiment
                     # mlbasics_learn_to_add
                     # mlbasics_rnn_binary_addition
                     # sparse_lstm_sentiment
                     # sparse_ner
                     # sparse_paraphrase
                     mnist_training
                     small_fc
                     bert_training
                     sentiment_neuron_training
                     onnx_load
                     # visualizer
                     )
message("DALI_AND_DEPS_LIBRARIES => " ${DALI_AND_DEPS_LIBRARIES})
configure_file("${PROJECT_SOURCE_DIR}/examples/config.h.in" "${PROJECT_SOURCE_DIR}/examples/config.h")
foreach(program ${EXAMPLE_PROGRAMS})
    add_executable(${program} ${EXAMPLES_DIR}/${program}.cpp)
    target_link_libraries(${program} utils ${DALI_AND_DEPS_LIBRARIES} ${GFLAGS_LIBRARIES})

    # TODO(jonathan): fix this. preferably move to dali, or at least make
    # a cmake function provided by find dali
    # if (MKL_FOUND AND BLAS_FOUND)
    #     add_custom_command(TARGET ${program}
    #         COMMAND sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${EXAMPLES_BUILD_DIR}/${program}
    #         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    # endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)
