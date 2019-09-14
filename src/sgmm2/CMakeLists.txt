set(PUBLIC_HEADERS
    am-sgmm2-project.h
    am-sgmm2.h
    decodable-am-sgmm2.h
    estimate-am-sgmm2-ebw.h
    estimate-am-sgmm2.h
    fmllr-sgmm2.h
)

add_library(kaldi-sgmm2
    am-sgmm2-project.cc
    am-sgmm2.cc
    decodable-am-sgmm2.cc
    estimate-am-sgmm2-ebw.cc
    estimate-am-sgmm2.cc
    fmllr-sgmm2.cc
)

target_include_directories(kaldi-sgmm2 PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-sgmm2 PUBLIC
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME am-sgmm2-test SOURCES am-sgmm2-test.cc DEPENDS kaldi-sgmm2)
    add_kaldi_test_executable(NAME estimate-am-sgmm2-test SOURCES estimate-am-sgmm2-test.cc DEPENDS kaldi-sgmm2)
    add_kaldi_test_executable(NAME fmllr-sgmm2-test SOURCES fmllr-sgmm2-test.cc DEPENDS kaldi-sgmm2)
endif()

install(TARGETS kaldi-sgmm2
    EXPORT kaldi-sgmm2-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/sgmm2)

