set(PUBLIC_HEADERS
    arctic-weight.h
    compose-lattice-pruned.h
    confidence.h
    determinize-lattice-pruned.h
    kaldi-lattice.h
    lattice-functions.h
    minimize-lattice.h
    phone-align-lattice.h
    push-lattice.h
    sausages.h
    word-align-lattice-lexicon.h
    word-align-lattice.h
)

add_library(kaldi-lat
    compose-lattice-pruned.cc
    confidence.cc
    determinize-lattice-pruned.cc
    kaldi-lattice.cc
    lattice-functions.cc
    minimize-lattice.cc
    phone-align-lattice.cc
    push-lattice.cc
    sausages.cc
    word-align-lattice-lexicon.cc
    word-align-lattice.cc
)

target_include_directories(kaldi-lat PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-lat PUBLIC
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME determinize-lattice-pruned-test SOURCES determinize-lattice-pruned-test.cc DEPENDS kaldi-lat)
    add_kaldi_test_executable(NAME kaldi-lattice-test SOURCES kaldi-lattice-test.cc DEPENDS kaldi-lat)
    add_kaldi_test_executable(NAME minimize-lattice-test SOURCES minimize-lattice-test.cc DEPENDS kaldi-lat)
    add_kaldi_test_executable(NAME push-lattice-test SOURCES push-lattice-test.cc DEPENDS kaldi-lat)
    add_kaldi_test_executable(NAME word-align-lattice-lexicon-test SOURCES word-align-lattice-lexicon-test.cc DEPENDS kaldi-lat)
endif()

install(TARGETS kaldi-lat
    EXPORT kaldi-lat-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/lat)

