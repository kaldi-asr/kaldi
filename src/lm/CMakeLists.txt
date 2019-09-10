add_library(kaldi-lm
    mikolov-rnnlm-lib.cc
    arpa-lm-compiler.cc
    kaldi-rnnlm.cc
    arpa-file-parser.cc
    const-arpa-lm.cc
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
    add_kaldi_test_executable(NAME arpa-lm-compiler-test SOURCES arpa-lm-compiler-test.cc DEPENDS kaldi-lm)
    add_kaldi_test_executable(NAME arpa-file-parser-test SOURCES arpa-file-parser-test.cc DEPENDS kaldi-lm)
endif()
