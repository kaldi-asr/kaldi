set(PUBLIC_HEADERS
    feature-common-inl.h
    feature-common.h
    feature-fbank.h
    feature-functions.h
    feature-mfcc.h
    feature-plp.h
    feature-spectrogram.h
    feature-window.h
    mel-computations.h
    online-feature.h
    pitch-functions.h
    resample.h
    signal.h
    wave-reader.h
)

add_library(kaldi-feat
    feature-fbank.cc
    feature-functions.cc
    feature-mfcc.cc
    feature-plp.cc
    feature-spectrogram.cc
    feature-window.cc
    mel-computations.cc
    online-feature.cc
    pitch-functions.cc
    resample.cc
    signal.cc
    wave-reader.cc
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
    add_kaldi_test_executable(NAME feature-functions-test SOURCES feature-functions-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-mfcc-test SOURCES feature-mfcc-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-plp-test SOURCES feature-plp-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME feature-sdc-test SOURCES feature-sdc-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME online-feature-test SOURCES online-feature-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME pitch-functions-test SOURCES pitch-functions-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME resample-test SOURCES resample-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME signal-test SOURCES signal-test.cc DEPENDS kaldi-feat)
    add_kaldi_test_executable(NAME wave-reader-test SOURCES wave-reader-test.cc DEPENDS kaldi-feat)
endif()

install(TARGETS kaldi-feat
    EXPORT kaldi-feat-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/feat)

