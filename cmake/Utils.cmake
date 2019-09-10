macro(add_kaldi_executable)
    cmake_parse_arguments(kaldi_exe "" "NAME" "SOURCES;DEPENDS" ${ARGN})
    add_executable(${kaldi_exe_NAME} ${kaldi_exe_SOURCES})
    target_link_libraries(${kaldi_exe_NAME} PRIVATE ${kaldi_exe_DEPENDS})
    # list(APPEND KALDI_EXECUTABLES ${kaldi_exe_NAME})
    install(TARGETS ${kaldi_exe_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    unset(kaldi_exe_NAME)
    unset(kaldi_exe_SOURCES)
    unset(kaldi_exe_DEPENDS)
endmacro()



macro(add_kaldi_test_executable)
    cmake_parse_arguments(kaldi_test_exe "" "NAME" "SOURCES;DEPENDS" ${ARGN})
    add_executable(${kaldi_test_exe_NAME} ${kaldi_test_exe_SOURCES})
    target_link_libraries(${kaldi_test_exe_NAME} PRIVATE ${kaldi_test_exe_DEPENDS})
    # list(APPEND KALDI_TEST_EXECUTABLES ${kaldi_test_exe_NAME})
    install(TARGETS ${kaldi_test_exe_NAME} RUNTIME DESTINATION testbin)

    unset(kaldi_test_exe_NAME)
    unset(kaldi_test_exe_SOURCES)
    unset(kaldi_test_exe_DEPENDS)
endmacro()
