set(PUBLIC_HEADERS
    build-tree-questions.h
    build-tree-utils.h
    build-tree.h
    cluster-utils.h
    clusterable-classes.h
    context-dep.h
    event-map.h
    tree-renderer.h
)

add_library(kaldi-tree
    build-tree-questions.cc
    build-tree-utils.cc
    build-tree.cc
    cluster-utils.cc
    clusterable-classes.cc
    context-dep.cc
    event-map.cc
    tree-renderer.cc
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
    add_kaldi_test_executable(NAME build-tree-test SOURCES build-tree-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME build-tree-utils-test SOURCES build-tree-utils-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME cluster-utils-test SOURCES cluster-utils-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME context-dep-test SOURCES context-dep-test.cc DEPENDS kaldi-tree)
    add_kaldi_test_executable(NAME event-map-test SOURCES event-map-test.cc DEPENDS kaldi-tree)
endif()

install(TARGETS kaldi-tree
    EXPORT kaldi-tree-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/tree)

