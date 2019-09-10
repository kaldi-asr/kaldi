add_library(kaldi-tree
    build-tree.cc
    context-dep.cc
    build-tree-questions.cc
    cluster-utils.cc
    event-map.cc
    clusterable-classes.cc
    tree-renderer.cc
    build-tree-utils.cc
)

target_include_directories(kaldi-tree PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-tree PUBLIC
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME cluster-utils-test SOURCES cluster-utils-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME event-map-test SOURCES event-map-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME build-tree-utils-test SOURCES build-tree-utils-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME build-tree-test SOURCES build-tree-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME context-dep-test SOURCES context-dep-test.cc DEPENDS kaldi-tree)
endif()
