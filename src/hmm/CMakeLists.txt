set(PUBLIC_HEADERS
    posterior.h
    hmm-utils.h
    hmm-test-utils.h
    hmm-topology.h
    transition-model.h
    tree-accu.h
)

add_library(kaldi-hmm
    hmm-utils.cc
    transition-model.cc
    tree-accu.cc
    hmm-test-utils.cc
    posterior.cc
    hmm-topology.cc
)

target_include_directories(kaldi-hmm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-hmm PUBLIC
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME hmm-utils-test SOURCES hmm-utils-test.cc DEPENDS kaldi-hmm)
    add_kaldi_test_executable(NAME hmm-topology-test SOURCES hmm-topology-test.cc DEPENDS kaldi-hmm)
    add_kaldi_test_executable(NAME posterior-test SOURCES posterior-test.cc DEPENDS kaldi-hmm)
    add_kaldi_test_executable(NAME transition-model-test SOURCES transition-model-test.cc DEPENDS kaldi-hmm)
endif()

install(TARGETS kaldi-hmm
    EXPORT kaldi-hmm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/hmm)

