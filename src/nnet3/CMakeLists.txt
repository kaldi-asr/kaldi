set(PUBLIC_HEADERS
    am-nnet-simple.h
    attention.h
    convolution.h
    decodable-online-looped.h
    decodable-simple-looped.h
    discriminative-supervision.h
    discriminative-training.h
    natural-gradient-online.h
    nnet-am-decodable-simple.h
    nnet-analyze.h
    nnet-attention-component.h
    nnet-batch-compute.h
    nnet-chain-diagnostics.h
    nnet-chain-example.h
    nnet-chain-training.h
    nnet-combined-component.h
    nnet-common.h
    nnet-compile-looped.h
    nnet-compile-utils.h
    nnet-compile.h
    nnet-component-itf.h
    nnet-computation-graph.h
    nnet-computation.h
    nnet-compute.h
    nnet-convolutional-component.h
    nnet-descriptor.h
    nnet-diagnostics.h
    nnet-discriminative-diagnostics.h
    nnet-discriminative-example.h
    nnet-discriminative-training.h
    nnet-example-utils.h
    nnet-example.h
    nnet-general-component.h
    nnet-graph.h
    nnet-nnet.h
    nnet-normalize-component.h
    nnet-optimize-utils.h
    nnet-optimize.h
    nnet-parse.h
    nnet-simple-component.h
    nnet-test-utils.h
    nnet-training.h
    nnet-utils.h
)

add_library(kaldi-nnet3
    am-nnet-simple.cc
    attention.cc
    convolution.cc
    decodable-online-looped.cc
    decodable-simple-looped.cc
    discriminative-supervision.cc
    discriminative-training.cc
    natural-gradient-online.cc
    nnet-am-decodable-simple.cc
    nnet-analyze.cc
    nnet-attention-component.cc
    nnet-batch-compute.cc
    nnet-chain-diagnostics.cc
    nnet-chain-example.cc
    nnet-chain-training.cc
    nnet-combined-component.cc
    nnet-common.cc
    nnet-compile-looped.cc
    nnet-compile-utils.cc
    nnet-compile.cc
    nnet-component-itf.cc
    nnet-computation-graph.cc
    nnet-computation.cc
    nnet-compute.cc
    nnet-convolutional-component.cc
    nnet-descriptor.cc
    nnet-diagnostics.cc
    nnet-discriminative-diagnostics.cc
    nnet-discriminative-example.cc
    nnet-discriminative-training.cc
    nnet-example-utils.cc
    nnet-example.cc
    nnet-general-component.cc
    nnet-graph.cc
    nnet-nnet.cc
    nnet-normalize-component.cc
    nnet-optimize-utils.cc
    nnet-optimize.cc
    nnet-parse.cc
    nnet-simple-component.cc
    nnet-tdnn-component.cc
    nnet-test-utils.cc
    nnet-training.cc
    nnet-utils.cc
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
    add_kaldi_test_executable(NAME attention-test SOURCES attention-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME convolution-test SOURCES convolution-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME natural-gradient-online-test SOURCES natural-gradient-online-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-analyze-test SOURCES nnet-analyze-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-common-test SOURCES nnet-common-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compile-test SOURCES nnet-compile-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compile-utils-test SOURCES nnet-compile-utils-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-component-test SOURCES nnet-component-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-compute-test SOURCES nnet-compute-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-derivative-test SOURCES nnet-derivative-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-descriptor-test SOURCES nnet-descriptor-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-example-test SOURCES nnet-example-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-graph-test SOURCES nnet-graph-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-nnet-test SOURCES nnet-nnet-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-optimize-test SOURCES nnet-optimize-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-parse-test SOURCES nnet-parse-test.cc DEPENDS kaldi-nnet3)
    add_kaldi_test_executable(NAME nnet3-utils-test SOURCES nnet-utils-test.cc DEPENDS kaldi-nnet3)
endif()

install(TARGETS kaldi-nnet3
    EXPORT kaldi-nnet3-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/nnet3)

