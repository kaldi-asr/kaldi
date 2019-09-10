set(PUBLIC_HEADERS
    feature-window-cuda.h
    feature-online-cmvn-cuda.h
    online-cuda-feature-pipeline.h
    online-ivector-feature-cuda.h
    online-ivector-feature-cuda-kernels.h
    feature-spectral-cuda.h
)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    feature-window-cuda.cu
    feature-online-cmvn-cuda.cu
    online-ivector-feature-cuda-kernels.cu
    feature-spectral-cuda.cu
)

add_library(kaldi-cudafeat
    online-cuda-feature-pipeline.cc
    online-ivector-feature-cuda.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-cudafeat PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-cudafeat PUBLIC
    kaldi-feat
    kaldi-util
    kaldi-matrix
    kaldi-base
    kaldi-cudamatrix
    kaldi-gmm
    kaldi-ivector
    kaldi-online2
)


install(TARGETS kaldi-cudafeat
    EXPORT kaldi-cudafeat-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/cudafeat)

