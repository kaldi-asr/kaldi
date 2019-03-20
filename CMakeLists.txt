cmake_minimum_required(VERSION 3.7.2)

project(Kaldi VERSION 6.0.0 LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/tools/cmake" ${CMAKE_MODULE_PATH})

enable_testing()

# You probably want one of "OpenBLAS", "Apple", "ATLAS", or "Intel10_64lp_seq"
# Details: https://github.com/Kitware/CMake/blob/283330976f2379eec214bfe1d51a27c29cc1f8fc/Modules/FindLAPACK.cmake#L25
set(ENV{OpenBLAS_HOME} "/export/b02/ws15dgalvez/kaldi-workspace/kaldi/tools/OpenBLAS/install")
# set(BLA_VENDOR "OpenBLAS" CACHE STRING "")
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(OpenBLAS REQUIRED)
add_library(OpenBLAS SHARED IMPORTED)
set_property(TARGET OpenBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenBLAS_INCLUDE_DIR})
set_property(TARGET OpenBLAS PROPERTY IMPORTED_LOCATION ${OpenBLAS_LIB})
set_property(TARGET OpenBLAS PROPERTY INTERFACE_COMPILE_DEFINITIONS HAVE_OPENBLAS=1)

add_library(OpenFST SHARED IMPORTED)
set_property(TARGET OpenFST PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/tools/openfst/include)
set_property(TARGET OpenFST PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/tools/openfst/lib/libfst.so)
file(GLOB OpenFST_LIBRARIES ${PROJECT_SOURCE_DIR}/tools/openfst/lib/lib*.so ${PROJECT_SOURCE_DIR}/tools/openfst/lib/fst/lib*.so)
set_property(TARGET OpenFST PROPERTY INTERFACE_LINK_LIBRARIES ${OpenFST_LIBRARIES})

find_package(Threads REQUIRED)
# include("${PROJECT_SOURCE_DIR}/tools/external.cmake")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)
