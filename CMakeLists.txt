cmake_minimum_required(VERSION 3.7.2)

project(Kaldi VERSION 6.0.0 LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/tools/cmake" ${CMAKE_MODULE_PATH})

enable_testing()

# You probably want one of "OpenBLAS", "Apple", "ATLAS", or "Intel10_64lp_seq"
# Details: https://github.com/Kitware/CMake/blob/283330976f2379eec214bfe1d51a27c29cc1f8fc/Modules/FindLAPACK.cmake#L25
set(ENV{OpenBLAS_HOME} "/export/b02/ws15dgalvez/kaldi-workspace/kaldi/tools/OpenBLAS/install")
# set(BLA_VENDOR "OpenBLAS" CACHE STRING "")
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(Threads REQUIRED)
include("${PROJECT_SOURCE_DIR}/tools/external.cmake")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)
