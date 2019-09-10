add_library(kaldi-transform
    fmllr-raw.cc
    basis-fmllr-diag-gmm.cc
    lda-estimate.cc
    cmvn.cc
    fmllr-diag-gmm.cc
    lvtln.cc
    compressed-transform-stats.cc
    transform-common.cc
    regtree-fmllr-diag-gmm.cc
    decodable-am-diag-gmm-regtree.cc
    regression-tree.cc
    fmpe.cc
    mllt.cc
    regtree-mllr-diag-gmm.cc
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
    add_kaldi_test_executable(NAME regtree-fmllr-diag-gmm-test SOURCES regtree-fmllr-diag-gmm-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME fmpe-test SOURCES fmpe-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME regtree-mllr-diag-gmm-test SOURCES regtree-mllr-diag-gmm-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME fmllr-diag-gmm-test SOURCES fmllr-diag-gmm-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME regression-tree-test SOURCES regression-tree-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME lda-estimate-test SOURCES lda-estimate-test.cc DEPENDS kaldi-transform)
    add_kaldi_test_executable(NAME fmllr-raw-test SOURCES fmllr-raw-test.cc DEPENDS kaldi-transform)
endif()
