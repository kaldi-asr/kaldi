set(PUBLIC_HEADERS
    feature-common.h
    signal.h
    feature-common-inl.h
    feature-functions.h
    feature-plp.h
    mel-computations.h
    feature-window.h
    pitch-functions.h
    online-feature.h
    feature-fbank.h
    resample.h
    wave-reader.h
    feature-mfcc.h
    feature-spectrogram.h
)

add_library(kaldi-feat
    feature-plp.cc
    feature-window.cc
    pitch-functions.cc
    feature-fbank.cc
    wave-reader.cc
    resample.cc
    online-feature.cc
    feature-mfcc.cc
    feature-spectrogram.cc
    mel-computations.cc
    feature-functions.cc
    signal.cc
)

target_include_directories(kaldi-feat PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-feat PUBLIC
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME feature-fbank-test SOURCES feature-fbank-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME wave-reader-test SOURCES wave-reader-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME online-feature-test SOURCES online-feature-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME resample-test SOURCES resample-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-functions-test SOURCES feature-functions-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME signal-test SOURCES signal-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-sdc-test SOURCES feature-sdc-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME pitch-functions-test SOURCES pitch-functions-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-mfcc-test SOURCES feature-mfcc-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-plp-test SOURCES feature-plp-test.cc DEPENDS kaldi-feat)
endif()

install(TARGETS kaldi-feat
    EXPORT kaldi-feat-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/feat)

