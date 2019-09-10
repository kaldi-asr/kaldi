set(PUBLIC_HEADERS
    cblas-wrappers.h
    sp-matrix-inl.h
    sp-matrix.h
    srfft.h
    compressed-matrix.h
    matrix-common.h
    sparse-matrix.h
    jama-eig.h
    kaldi-matrix-inl.h
    tp-matrix.h
    kaldi-vector-inl.h
    kaldi-vector.h
    kaldi-blas.h
    jama-svd.h
    matrix-lib.h
    packed-matrix.h
    matrix-functions-inl.h
    matrix-functions.h
    optimization.h
    kaldi-matrix.h
)

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

install(TARGETS kaldi-matrix
    EXPORT kaldi-matrix-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/matrix)

