add_library(kaldi-matrix
    sp-matrix.cc
    packed-matrix.cc
    compressed-matrix.cc
    qr.cc
    kaldi-vector.cc
    srfft.cc
    sparse-matrix.cc
    matrix-functions.cc
    kaldi-matrix.cc
    optimization.cc
    tp-matrix.cc
)

target_include_directories(kaldi-matrix PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-matrix PUBLIC
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME matrix-lib-speed-test SOURCES matrix-lib-speed-test.cc DEPENDS kaldi-matrix)
    add_kaldi_test_executable(NAME matrix-lib-test SOURCES matrix-lib-test.cc DEPENDS kaldi-matrix)
    add_kaldi_test_executable(NAME sparse-matrix-test SOURCES sparse-matrix-test.cc DEPENDS kaldi-matrix)
endif()
