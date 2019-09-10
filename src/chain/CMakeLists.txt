cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    chain-kernels.cu
)

add_library(kaldi-chain
    chain-generic-numerator.cc
    chain-denominator.cc
    chain-training.cc
    chain-den-graph.cc
    chain-numerator.cc
    language-model.cc
    chain-supervision.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-chain PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-chain PUBLIC
    kaldi-cudamatrix
    kaldi-lat
    kaldi-fstext
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME language-model-test SOURCES language-model-test.cc DEPENDS kaldi-chain)
    add_kaldi_test_executable(NAME chain-supervision-test SOURCES chain-supervision-test.cc DEPENDS kaldi-chain)
endif()
