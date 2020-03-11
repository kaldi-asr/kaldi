if(NOT CMAKE_VERSION VERSION_LESS "3.10")
    include_guard()
endif()

# For Windows, some env or vars are using backward slash for pathes, convert
# them to forward slashes will fix some nasty problem in CMake.
macro(normalize_path in_path)
    file(TO_CMAKE_PATH "${${in_path}}" normalize_path_out_path)
    set(${in_path} "${normalize_path_out_path}")
    unset(normalize_path_out_path)
endmacro()

macro(normalize_env_path in_path)
    file(TO_CMAKE_PATH "$${in_path}" normalize_env_path_out_path)
    set(${in_path} "${normalize_env_path_out_path}")
    unset(normalize_env_path_out_path)
endmacro()


macro(add_kaldi_executable)
    if(${KALDI_BUILD_EXE})
        cmake_parse_arguments(kaldi_exe "" "NAME" "SOURCES;DEPENDS" ${ARGN})
        add_executable(${kaldi_exe_NAME} ${kaldi_exe_SOURCES})
        target_link_libraries(${kaldi_exe_NAME} PRIVATE ${kaldi_exe_DEPENDS})
        # list(APPEND KALDI_EXECUTABLES ${kaldi_exe_NAME})
        install(TARGETS ${kaldi_exe_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

        unset(kaldi_exe_NAME)
        unset(kaldi_exe_SOURCES)
        unset(kaldi_exe_DEPENDS)
    endif()
endmacro()

macro(add_kaldi_test_executable)
    if(${KALDI_BUILD_TEST})
        cmake_parse_arguments(kaldi_test_exe "" "NAME" "SOURCES;DEPENDS" ${ARGN})
        add_executable(${kaldi_test_exe_NAME} ${kaldi_test_exe_SOURCES})
        target_link_libraries(${kaldi_test_exe_NAME} PRIVATE ${kaldi_test_exe_DEPENDS})
        add_test(
            NAME ${kaldi_test_exe_NAME}
            COMMAND ${kaldi_test_exe_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
        # list(APPEND KALDI_TEST_EXECUTABLES ${kaldi_test_exe_NAME})
        install(TARGETS ${kaldi_test_exe_NAME} RUNTIME DESTINATION testbin)

        unset(kaldi_test_exe_NAME)
        unset(kaldi_test_exe_SOURCES)
        unset(kaldi_test_exe_DEPENDS)
    endif()
endmacro()
