cmake_minimum_required(VERSION 3.7.2)

project(Kaldi VERSION 6.0.0 LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/tools/cmake" ${CMAKE_MODULE_PATH})

include("${PROJECT_SOURCE_DIR}/tools/external.cmake")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)
