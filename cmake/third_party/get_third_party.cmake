# Download and unpack a third-party library at configure time
# The original code is at the README of google-test:
# https://github.com/google/googletest/tree/master/googletest
function(get_third_party name)
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/third_party/${name}.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${name}-download/CMakeLists.txt")
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${name}-download")
    if(result)
        message(FATAL_ERROR "CMake step for ${name} failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${name}-download")
    if(result)
        message(FATAL_ERROR "Build step for ${name} failed: ${result}")
    endif()
endfunction()
