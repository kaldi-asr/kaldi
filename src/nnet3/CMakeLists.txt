add_library(kaldi-nnet3
    nnet-chain-diagnostics.cc
    nnet-discriminative-diagnostics.cc
    nnet-computation.cc
    nnet-compile.cc
    nnet-analyze.cc
    nnet-compile-utils.cc
    nnet-optimize-utils.cc
    nnet-example.cc
    attention.cc
    nnet-discriminative-training.cc
    nnet-simple-component.cc
    nnet-attention-component.cc
    nnet-example-utils.cc
    nnet-component-itf.cc
    nnet-nnet.cc
    nnet-convolutional-component.cc
    nnet-am-decodable-simple.cc
    nnet-compute.cc
    discriminative-training.cc
    nnet-computation-graph.cc
    convolution.cc
    nnet-optimize.cc
    nnet-combined-component.cc
    discriminative-supervision.cc
    nnet-utils.cc
    nnet-normalize-component.cc
    nnet-chain-example.cc
    nnet-compile-looped.cc
    natural-gradient-online.cc
    am-nnet-simple.cc
    nnet-descriptor.cc
    nnet-parse.cc
    nnet-tdnn-component.cc
    nnet-general-component.cc
    nnet-training.cc
    nnet-discriminative-example.cc
    decodable-online-looped.cc
    decodable-simple-looped.cc
    nnet-common.cc
    nnet-chain-training.cc
    nnet-batch-compute.cc
    nnet-diagnostics.cc
    nnet-test-utils.cc
    nnet-graph.cc
)

target_include_directories(kaldi-nnet3 PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-nnet3 PUBLIC
    kaldi-chain
    kaldi-cudamatrix
    kaldi-decoder
    kaldi-lat
    kaldi-fstext
    kaldi-hmm
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME nnet3-graph-test SOURCES nnet-graph-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-optimize-test SOURCES nnet-optimize-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compute-test SOURCES nnet-compute-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME natural-gradient-online-test SOURCES natural-gradient-online-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-common-test SOURCES nnet-common-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-utils-test SOURCES nnet-utils-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-descriptor-test SOURCES nnet-descriptor-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-derivative-test SOURCES nnet-derivative-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compile-utils-test SOURCES nnet-compile-utils-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-analyze-test SOURCES nnet-analyze-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-nnet-test SOURCES nnet-nnet-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-parse-test SOURCES nnet-parse-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME attention-test SOURCES attention-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-example-test SOURCES nnet-example-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compile-test SOURCES nnet-compile-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME convolution-test SOURCES convolution-test.cc DEPENDS kaldi-nnet3)
endif()
