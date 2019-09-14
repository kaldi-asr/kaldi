set(PUBLIC_HEADERS
    am-diag-gmm.h
    decodable-am-diag-gmm.h
    diag-gmm-inl.h
    diag-gmm-normal.h
    diag-gmm.h
    ebw-diag-gmm.h
    full-gmm-inl.h
    full-gmm-normal.h
    full-gmm.h
    indirect-diff-diag-gmm.h
    mle-am-diag-gmm.h
    mle-diag-gmm.h
    mle-full-gmm.h
    model-common.h
    model-test-common.h
)

add_library(kaldi-gmm
    am-diag-gmm.cc
    decodable-am-diag-gmm.cc
    diag-gmm-normal.cc
    diag-gmm.cc
    ebw-diag-gmm.cc
    full-gmm-normal.cc
    full-gmm.cc
    indirect-diff-diag-gmm.cc
    mle-am-diag-gmm.cc
    mle-diag-gmm.cc
    mle-full-gmm.cc
    model-common.cc
    model-test-common.cc
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
    add_kaldi_test_executable(NAME am-diag-gmm-test SOURCES am-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME diag-gmm-test SOURCES diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME ebw-diag-gmm-test SOURCES ebw-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME full-gmm-test SOURCES full-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-am-diag-gmm-test SOURCES mle-am-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-diag-gmm-test SOURCES mle-diag-gmm-test.cc DEPENDS kaldi-gmm)
    add_kaldi_test_executable(NAME mle-full-gmm-test SOURCES mle-full-gmm-test.cc DEPENDS kaldi-gmm)
endif()

install(TARGETS kaldi-gmm
    EXPORT kaldi-gmm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/gmm)

