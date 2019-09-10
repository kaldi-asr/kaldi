set(PUBLIC_HEADERS
    diag-gmm-inl.h
    indirect-diff-diag-gmm.h
    am-diag-gmm.h
    model-test-common.h
    full-gmm-normal.h
    ebw-diag-gmm.h
    diag-gmm-normal.h
    mle-full-gmm.h
    full-gmm.h
    mle-diag-gmm.h
    diag-gmm.h
    mle-am-diag-gmm.h
    full-gmm-inl.h
    model-common.h
    decodable-am-diag-gmm.h
)

add_library(kaldi-gmm
    indirect-diff-diag-gmm.cc
    full-gmm-normal.cc
    model-test-common.cc
    mle-diag-gmm.cc
    mle-am-diag-gmm.cc
    model-common.cc
    mle-full-gmm.cc
    ebw-diag-gmm.cc
    full-gmm.cc
    am-diag-gmm.cc
    diag-gmm-normal.cc
    decodable-am-diag-gmm.cc
    diag-gmm.cc
)

target_include_directories(kaldi-gmm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-gmm PUBLIC
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME diag-gmm-test SOURCES diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME ebw-diag-gmm-test SOURCES ebw-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-diag-gmm-test SOURCES mle-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME full-gmm-test SOURCES full-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-am-diag-gmm-test SOURCES mle-am-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-full-gmm-test SOURCES mle-full-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME am-diag-gmm-test SOURCES am-diag-gmm-test.cc DEPENDS kaldi-gmm)
endif()

install(TARGETS kaldi-gmm
    EXPORT kaldi-gmm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/gmm)

