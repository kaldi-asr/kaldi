cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_compile(CUDA_OBJS
    cu-kernels.cu
)

add_library(kaldi-cudamatrix
    cu-compressed-matrix.cc
    cu-vector.cc
    cu-tp-matrix.cc
    cu-device.cc
    cu-matrix.cc
    cu-common.cc
    cu-allocator.cc
    cu-sparse-matrix.cc
    cu-block-matrix.cc
    cu-array.cc
    cu-packed-matrix.cc
    cu-math.cc
    cu-rand.cc
    cu-sp-matrix.cc
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
    add_kaldi_test_executable(NAME cu-sparse-matrix-test SOURCES cu-sparse-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-matrix-test SOURCES cu-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-test SOURCES cu-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-packed-matrix-test SOURCES cu-packed-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-rand-speed-test SOURCES cu-rand-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-matrix-speed-test SOURCES cu-matrix-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-compressed-matrix-test SOURCES cu-compressed-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-math-test SOURCES cu-math-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-sp-matrix-speed-test SOURCES cu-sp-matrix-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-vector-test SOURCES cu-vector-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-sp-matrix-test SOURCES cu-sp-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-vector-speed-test SOURCES cu-vector-speed-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-tp-matrix-test SOURCES cu-tp-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-block-matrix-test SOURCES cu-block-matrix-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-device-test SOURCES cu-device-test.cc DEPENDS kaldi-cudamatrix)
    add_kaldi_test_executable(NAME cu-array-test SOURCES cu-array-test.cc DEPENDS kaldi-cudamatrix)
endif()
