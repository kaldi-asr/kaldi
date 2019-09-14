set(PUBLIC_HEADERS
    batched-threaded-nnet3-cuda-pipeline.h
    cuda-decodable-itf.h
    cuda-decoder-common.h
    cuda-decoder-kernels-utils.h
    cuda-decoder-kernels.h
    cuda-decoder.h
    cuda-fst.h
    decodable-cumatrix.h
    thread-pool.h
)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    cuda-decoder-kernels.cu
)

add_library(kaldi-cudadecoder
    batched-threaded-nnet3-cuda-pipeline.cc
    cuda-decoder.cc
    cuda-fst.cc
    decodable-cumatrix.cc
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


install(TARGETS kaldi-cudadecoder
    EXPORT kaldi-cudadecoder-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/cudadecoder)

