set(PUBLIC_HEADERS
    kaldi-utils.h
    kaldi-common.h
    timer.h
    kaldi-math.h
    io-funcs.h
    kaldi-types.h
    io-funcs-inl.h
    kaldi-error.h
)

add_library(kaldi-base
    kaldi-error.cc
    io-funcs.cc
    kaldi-utils.cc
    kaldi-math.cc
    timer.cc
)

target_include_directories(kaldi-base PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME io-funcs-test SOURCES io-funcs-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME timer-test SOURCES timer-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME kaldi-error-test SOURCES kaldi-error-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME kaldi-math-test SOURCES kaldi-math-test.cc DEPENDS kaldi-base)
endif()

install(TARGETS kaldi-base
    EXPORT kaldi-base-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/base)

