set(PUBLIC_HEADERS
    arpa-file-parser.h
    arpa-lm-compiler.h
    const-arpa-lm.h
    kaldi-rnnlm.h
    mikolov-rnnlm-lib.h
)

add_library(kaldi-lm
    arpa-file-parser.cc
    arpa-lm-compiler.cc
    const-arpa-lm.cc
    kaldi-rnnlm.cc
    mikolov-rnnlm-lib.cc
)

target_include_directories(kaldi-lm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-lm PUBLIC
    kaldi-fstext
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME arpa-file-parser-test SOURCES arpa-file-parser-test.cc DEPENDS kaldi-lm)
    add_kaldi_test_executable(NAME arpa-lm-compiler-test SOURCES arpa-lm-compiler-test.cc DEPENDS kaldi-lm)
endif()

install(TARGETS kaldi-lm
    EXPORT kaldi-lm-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/lm)

