set(PUBLIC_HEADERS
    cu-allocator.h
    cu-array-inl.h
    cu-array.h
    cu-block-matrix.h
    cu-common.h
    cu-compressed-matrix.h
    cu-device.h
    cu-kernels-ansi.h
    cu-kernels.h
    cu-math.h
    cu-matrix-inl.h
    cu-matrix-lib.h
    cu-matrix.h
    cu-matrixdim.h
    cu-packed-matrix.h
    cu-rand.h
    cu-sp-matrix.h
    cu-sparse-matrix.h
    cu-tp-matrix.h
    cu-value.h
    cu-vector.h
    cublas-wrappers.h
)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    cu-kernels.cu
)

add_library(kaldi-cudamatrix
    cu-allocator.cc
    cu-array.cc
    cu-block-matrix.cc
    cu-common.cc
    cu-compressed-matrix.cc
    cu-device.cc
    cu-math.cc
    cu-matrix.cc
    cu-packed-matrix.cc
    cu-rand.cc
    cu-sp-matrix.cc
    cu-sparse-matrix.cc
    cu-tp-matrix.cc
    cu-vector.cc
    ${CUDA_OBJS}
)

target_include_directories(kaldi-cudamatrix PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-cudamatrix PUBLIC
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME cu-array-test SOURCES cu-array-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-block-matrix-test SOURCES cu-block-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-compressed-matrix-test SOURCES cu-compressed-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-device-test SOURCES cu-device-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-math-test SOURCES cu-math-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-matrix-speed-test SOURCES cu-matrix-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-matrix-test SOURCES cu-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-packed-matrix-test SOURCES cu-packed-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-rand-speed-test SOURCES cu-rand-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-sp-matrix-speed-test SOURCES cu-sp-matrix-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-sp-matrix-test SOURCES cu-sp-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-sparse-matrix-test SOURCES cu-sparse-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-test SOURCES cu-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-tp-matrix-test SOURCES cu-tp-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-vector-speed-test SOURCES cu-vector-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-vector-test SOURCES cu-vector-test.cc DEPENDS kaldi-cudamatrix)
endif()

install(TARGETS kaldi-cudamatrix
    EXPORT kaldi-cudamatrix-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/cudamatrix)

