cmake_minimum_required(VERSION 3.5)
project(kaldi)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(Utils)

set(MATHLIB "OpenBLAS" CACHE STRING "OpenBLAS|MKL")
option(KALDI_BUILD_TEST "" ON)

set(CMAKE_CXX_STANDARD 14)

# TODO: fix this
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../openfst-winport/src/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../openfst-winport/build_output/x64/Debug/lib)
link_libraries(libfst.lib)


if(MATHLIB STREQUAL "OpenBLAS")
    set(BLA_VENDOR "OpenBLAS")
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_CLAPACK=1)
elseif(MATHLIB STREQUAL "MKL")
    set(BLA_VENDOR "Intel10_64lp")
    # find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_MKL=1)
    include_directories($ENV{MKLROOT}/include) # TODO: maybe not use env, idk, find_package doesnt handle includes...
    link_libraries(${BLAS_LIBRARIES} "${LAPACK_LIBRARIES}")
else()
    message(FATAL_ERROR "TODO: error message" "${MATHLIB}")
endif()

find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DHAVE_CUDA=1)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(${CUDA_LIBRARY})

    find_package(NvToolExt REQUIRED)
    link_libraries(NvToolExt)
endif()

# TODO: version helper
add_definitions(-DKALDI_VERSION="cmake")

# add all native libraries
add_subdirectory(src/base)
add_subdirectory(src/matrix)
add_subdirectory(src/util)
add_subdirectory(src/tree)
add_subdirectory(src/gmm)
add_subdirectory(src/transform)
add_subdirectory(src/feat)

# add all cuda libraries
if(CUDA_FOUND)
    add_subdirectory(src/cudamatrix)
    add_subdirectory(src/cudafeat)
endif()

# add all native executables
add_subdirectory(src/gmmbin)
add_subdirectory(src/featbin)

# add all cuda executables
if(CUDA_FOUND)
    add_subdirectory(src/cudafeatbin)
endif()
