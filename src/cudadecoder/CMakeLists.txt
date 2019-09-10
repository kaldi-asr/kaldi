cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    cuda-decoder-kernels.cu
)

add_library(kaldi-cudadecoder
    cuda-fst.cc
    decodable-cumatrix.cc
    batched-threaded-nnet3-cuda-pipeline.cc
    cuda-decoder.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-cudadecoder PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-cudadecoder PUBLIC
    kaldi-cudamatrix
    kaldi-base
    kaldi-matrix
    kaldi-lat
    kaldi-util
    kaldi-matrix
    kaldi-gmm
    kaldi-fstext
    kaldi-hmm
    kaldi-gmm
    kaldi-transform
    kaldi-tree
    kaldi-online2
    kaldi-nnet3
    kaldi-cudafeat
)

