set(PUBLIC_HEADERS
    io-funcs-inl.h
    io-funcs.h
    kaldi-common.h
    kaldi-error.h
    kaldi-math.h
    kaldi-types.h
    kaldi-utils.h
    timer.h
)

add_library(kaldi-base
    io-funcs.cc
    kaldi-error.cc
    kaldi-math.cc
    kaldi-utils.cc
    timer.cc
)

target_include_directories(kaldi-base PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

if(KALDI_BUILD_TEST)
    add_kaldi_test_executable(NAME io-funcs-test SOURCES io-funcs-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME kaldi-error-test SOURCES kaldi-error-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME kaldi-math-test SOURCES kaldi-math-test.cc DEPENDS kaldi-base)
    add_kaldi_test_executable(NAME timer-test SOURCES timer-test.cc DEPENDS kaldi-base)
endif()

install(TARGETS kaldi-base
    EXPORT kaldi-base-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/base)

