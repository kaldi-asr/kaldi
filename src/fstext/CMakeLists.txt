set(PUBLIC_HEADERS
    context-fst.h
    deterministic-fst-inl.h
    deterministic-fst.h
    determinize-lattice-inl.h
    determinize-lattice.h
    determinize-star-inl.h
    determinize-star.h
    epsilon-property-inl.h
    epsilon-property.h
    factor-inl.h
    factor.h
    fst-test-utils.h
    fstext-lib.h
    fstext-utils-inl.h
    fstext-utils.h
    grammar-context-fst.h
    kaldi-fst-io-inl.h
    kaldi-fst-io.h
    lattice-utils-inl.h
    lattice-utils.h
    lattice-weight.h
    pre-determinize-inl.h
    pre-determinize.h
    prune-special-inl.h
    prune-special.h
    push-special.h
    rand-fst.h
    remove-eps-local-inl.h
    remove-eps-local.h
    table-matcher.h
    trivial-factor-weight.h
)

add_library(kaldi-fstext
    context-fst.cc
    grammar-context-fst.cc
    kaldi-fst-io.cc
    push-special.cc
)

target_include_directories(kaldi-fstext PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-fstext PUBLIC
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME context-fst-test SOURCES context-fst-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME deterministic-fst-test SOURCES deterministic-fst-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME determinize-lattice-test SOURCES determinize-lattice-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME determinize-star-test SOURCES determinize-star-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME epsilon-property-test SOURCES epsilon-property-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME factor-test SOURCES factor-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME fstext-utils-test SOURCES fstext-utils-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME lattice-utils-test SOURCES lattice-utils-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME lattice-weight-test SOURCES lattice-weight-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME pre-determinize-test SOURCES pre-determinize-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME prune-special-test SOURCES prune-special-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME push-special-test SOURCES push-special-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME remove-eps-local-test SOURCES remove-eps-local-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME table-matcher-test SOURCES table-matcher-test.cc DEPENDS kaldi-fstext)
    add_kaldi_test_executable(NAME trivial-factor-weight-test SOURCES trivial-factor-weight-test.cc DEPENDS kaldi-fstext)
endif()

install(TARGETS kaldi-fstext
    EXPORT kaldi-fstext-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/fstext)

