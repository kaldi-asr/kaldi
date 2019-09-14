set(PUBLIC_HEADERS
    basis-fmllr-diag-gmm.h
    cmvn.h
    compressed-transform-stats.h
    decodable-am-diag-gmm-regtree.h
    fmllr-diag-gmm.h
    fmllr-raw.h
    fmpe.h
    lda-estimate.h
    lvtln.h
    mllt.h
    regression-tree.h
    regtree-fmllr-diag-gmm.h
    regtree-mllr-diag-gmm.h
    transform-common.h
)

add_library(kaldi-transform
    basis-fmllr-diag-gmm.cc
    cmvn.cc
    compressed-transform-stats.cc
    decodable-am-diag-gmm-regtree.cc
    fmllr-diag-gmm.cc
    fmllr-raw.cc
    fmpe.cc
    lda-estimate.cc
    lvtln.cc
    mllt.cc
    regression-tree.cc
    regtree-fmllr-diag-gmm.cc
    regtree-mllr-diag-gmm.cc
    transform-common.cc
)

target_include_directories(kaldi-transform PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-transform PUBLIC
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME fmllr-diag-gmm-test SOURCES fmllr-diag-gmm-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME fmllr-raw-test SOURCES fmllr-raw-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME fmpe-test SOURCES fmpe-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME lda-estimate-test SOURCES lda-estimate-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME regression-tree-test SOURCES regression-tree-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME regtree-fmllr-diag-gmm-test SOURCES regtree-fmllr-diag-gmm-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME regtree-mllr-diag-gmm-test SOURCES regtree-mllr-diag-gmm-test.cc DEPENDS kaldi-transform)
endif()

install(TARGETS kaldi-transform
    EXPORT kaldi-transform-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/transform)

