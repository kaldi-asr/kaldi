add_library(kaldi-online
    online-feat-input.cc
    online-decodable.cc
    online-faster-decoder.cc
    online-audio-source.cc
    online-tcp-source.cc
    onlinebin-util.cc
)

target_include_directories(kaldi-online PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-online PUBLIC
    kaldi-decoder
    kaldi-lat
    kaldi-hmm
    kaldi-feat
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME online-feat-test SOURCES online-feat-test.cc DEPENDS kaldi-online)
endif()
