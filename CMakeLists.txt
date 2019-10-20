cmake_minimum_required(VERSION 3.5)
project(kaldi)

execute_process(COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_cmake_skeleton.py" "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_MESSAGE LAZY) # hide "-- Up-to-date: ..."


set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(Utils)

set(MATHLIB "OpenBLAS" CACHE STRING "OpenBLAS|MKL|Accelerate")
option(KALDI_BUILD_TEST "" ON)

# TODO: fix this
if(WIN32)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../openfst-winport/src/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../openfst-winport/build_output/x64/Debug/lib)
    link_libraries(libfst.lib)
    add_compile_options(/bigobj)
else()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tools/openfst/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/tools/openfst/lib)
    link_libraries(fst)
endif()

link_libraries(${CMAKE_DL_LIBS})

find_package(Threads)
link_libraries(Threads::Threads)

if(MATHLIB STREQUAL "OpenBLAS")
    set(BLA_VENDOR "OpenBLAS")
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_CLAPACK=1)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tools/CLAPACK) # TODO:
    link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
elseif(MATHLIB STREQUAL "MKL")
    set(BLA_VENDOR "Intel10_64lp")
    # find_package(BLAS REQUIRED)
    normalize_env_path(ENV{MKLROOT})
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_MKL=1)
    include_directories($ENV{MKLROOT}/include) # TODO: maybe not use env, idk, find_package doesnt handle includes...
    link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
elseif(MATHLIB STREQUAL "Accelerate")
    set(BLA_VENDOR "Apple")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_CLAPACK=1)
    link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    message(FATAL_ERROR "${MATHLIB} is not tested and supported, you are on your own now.")
endif()

if(MSVC)
    # Added in source, but we actually should do it in build script, whatever...
    # add_definitions(-DWIN32_LEAN_AND_MEAN=1)

    add_compile_options(/permissive- /FS /wd4819)

    # some warnings related with fst
    add_compile_options(/wd4018 /wd4244 /wd4267 /wd4291 /wd4305)

    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
    if(NOT DEFINED ENV{CUDAHOSTCXX})
        set(ENV{CUDAHOSTCXX} ${CMAKE_CXX_COMPILER})
    endif()
    if(NOT DEFINED CUDA_HOST_COMPILER)
        set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
endif()

find_package(CUDA)
if(CUDA_FOUND)
    set(CUB_ROOT_DIR "tools/cub")

    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(KALDI_CUDA_NVCC_FLAGS "--default-stream=per-thread;-std=c++${CMAKE_CXX_STANDARD}")
    if(MSVC)
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /permissive-,/wd4819")
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /wd4018,/wd4244,/wd4267,/wd4291,/wd4305")
    else()
    #     list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler -std=c++${CMAKE_CXX_STANDARD}")
    endif()

    set(CUDA_NVCC_FLAGS ${KALDI_CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS})

    add_definitions(-DHAVE_CUDA=1)
    add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_libraries(
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cusparse_LIBRARY})

    find_package(NvToolExt REQUIRED)
    include_directories(${NvToolExt_INCLUDE_DIR})
    link_libraries(NvToolExt)

    find_package(CUB REQUIRED)
    include_directories(${CUB_INCLUDE_DIR})
endif()

# TODO: version helper
set(KALDI_VERSION 0.0)
add_definitions(-DKALDI_VERSION="${KALDI_VERSION}")
add_definitions(-DKALDI_NO_PORTAUDIO=1)

# add all native libraries
add_subdirectory(src/base)
add_subdirectory(src/matrix)
add_subdirectory(src/util)
add_subdirectory(src/feat)
add_subdirectory(src/tree)
add_subdirectory(src/gmm)
add_subdirectory(src/transform)
add_subdirectory(src/sgmm2)
add_subdirectory(src/fstext)
add_subdirectory(src/hmm)
add_subdirectory(src/lm)
add_subdirectory(src/decoder)
add_subdirectory(src/lat)
add_subdirectory(src/nnet)
add_subdirectory(src/nnet2)
add_subdirectory(src/nnet3)
add_subdirectory(src/rnnlm)
add_subdirectory(src/chain)
add_subdirectory(src/ivector)
add_subdirectory(src/online)
add_subdirectory(src/online2)
add_subdirectory(src/kws)

# add all cuda libraries
if(CUDA_FOUND)
    add_subdirectory(src/cudamatrix)
    add_subdirectory(src/cudafeat)
    add_subdirectory(src/cudadecoder)
endif()

# add all native executables
add_subdirectory(src/gmmbin)
add_subdirectory(src/featbin)
add_subdirectory(src/onlinebin)

# add all cuda executables
if(CUDA_FOUND)
    add_subdirectory(src/cudafeatbin)
    add_subdirectory(src/cudadecoderbin)
endif()

include(CMakePackageConfigHelpers)
# maybe we should put this into subfolder?
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/kaldi-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/kaldi-config.cmake
    INSTALL_DESTINATION lib/cmake/kaldi
)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/kaldi-config-version.cmake
    VERSION ${KALDI_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES ${CMAKE_BINARY_DIR}/cmake/kaldi-config.cmake ${CMAKE_BINARY_DIR}/cmake/kaldi-config-version.cmake
    DESTINATION lib/cmake/kaldi
)
