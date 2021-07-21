cmake_minimum_required(VERSION 3.5)
project(kaldi)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Install path prefix." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(GNUInstallDirs)
include(Utils)

include(third_party/get_third_party)

include(cmake/third_party/openfst.cmake)

find_package(PythonInterp)
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Needs python to auto-generate most CMake files, but not found.")
endif()

message(STATUS "Running gen_cmake_skeleton.py")
option(BUILD_SHARED_LIBS "Build shared Kaldi libraries." OFF)
set(IS_LIB_SHARE "")
if(BUILD_SHARED_LIBS)
    set(IS_LIB_SHARE "--shared")
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_cmake_skeleton.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "--quiet"
    ${IS_LIB_SHARE}
)
unset(IS_LIB_SHARE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_MESSAGE LAZY) # hide "-- Up-to-date: ..."
if(BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(WIN32)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        message(FATAL_ERROR "DLL is not supported currently")
    elseif(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../../tools/openfst/lib")
    endif()
endif()

if(APPLE)
    # Use built-in BLAS on MacOS by default.
    set(MATHLIB "Accelerate" CACHE STRING "OpenBLAS|MKL|Accelerate")
else()
    set(MATHLIB "OpenBLAS" CACHE STRING "OpenBLAS|MKL|Accelerate")
endif()
option(KALDI_BUILD_EXE "If disabled, will make add_kaldi_executable a no-op" ON)
option(KALDI_BUILD_TEST "If disabled, will make add_kaldi_test_executable a no-op" ON)
option(KALDI_USE_PATCH_NUMBER "Use MAJOR.MINOR.PATCH format, otherwise MAJOR.MINOR" OFF)

if (KALDI_BUILD_TEST)
    include(CTest)
    enable_testing()
endif()

link_libraries(${CMAKE_DL_LIBS})

find_package(Threads)
link_libraries(Threads::Threads)

if(MATHLIB STREQUAL "OpenBLAS")
    set(BLA_VENDOR "OpenBLAS")
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_CLAPACK=1)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tools/CLAPACK)
    link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
elseif(MATHLIB STREQUAL "MKL")
    if(NOT DEFINED ENV{MKLROOT} OR "$ENV{MKLROOT}" STREQUAL "")
        message(FATAL_ERROR "Environment variable MKLROOT is not defined")
    else()
        message(STATUS "Finding MKL from \"$ENV{MKLROOT}\"")
    endif()
    normalize_env_path(ENV{MKLROOT})
    set(BLA_VENDOR "Intel10_64lp_seq") # use the single threaded MKL by default
    find_package(LAPACK REQUIRED)
    add_definitions(-DHAVE_MKL=1)
    include_directories($ENV{MKLROOT}/include)
    link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
elseif(MATHLIB STREQUAL "Accelerate")
    execute_process(COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_VERSION)
    if(MACOS_VERSION VERSION_LESS "10.12" AND MACOS_VERSION VERSION_GREATER_EQUAL "10.11")
        message(WARNING
            "**BAD WARNING**: You are using OS X El Capitan.  Some versions of this OS"
            " have a bug in the BLAS implementation that affects Kaldi."
            " After compiling, cd to matrix/ and type 'make test'.  The"
            " test will fail if the problem exists in your version."
            " Eventually this issue will be fixed by system updates from"
            " Apple.  Unexplained crashes with reports of NaNs will"
            " be caused by this bug, but some recipes will (sometimes) work."
        )
    endif()
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

    add_compile_options(/permissive- /FS /wd4819 /EHsc /bigobj)

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
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(KALDI_CUDA_NVCC_FLAGS "--default-stream=per-thread;-std=c++${CMAKE_CXX_STANDARD}")
    if(MSVC)
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /permissive-,/FS,/wd4819,/EHsc,/bigobj")
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler /wd4018,/wd4244,/wd4267,/wd4291,/wd4305")
        if(BUILD_SHARED_LIBS)
            list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler /MD)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler /MDd)
        endif()
    else()
    #     list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler -std=c++${CMAKE_CXX_STANDARD}")
        list(APPEND KALDI_CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
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
    link_libraries(${NvToolExt_LIBRARIES})

    get_third_party(cub)
    set(CUB_ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub")
    find_package(CUB REQUIRED)
    include_directories(${CUB_INCLUDE_DIR})
endif()

add_definitions(-DKALDI_NO_PORTAUDIO=1)

include(VersionHelper)
get_version() # this will set KALDI_VERSION and KALDI_PATCH_NUMBER
if(${KALDI_USE_PATCH_NUMBER})
    set(KALDI_VERSION "${KALDI_VERSION}.${KALDI_PATCH_NUMBER}")
endif()

# get_third_party(openfst)
# set(OPENFST_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/openfst)
# include(third_party/openfst_lib_target)
#find_library(OpenFST_LIBRARY
#            NAMES fst
#            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/tools/openfst/lib
#            REQUIRED)
#find_path(OpenFST_INCLUDE_DIR
#            NAMES "fst/fst.h"
#            PATHS "${CMAKE_CURRENT_SOURCE_DIR}/tools/openfst/include"
#            REQUIRED)

link_libraries(fst)

# add all native libraries
add_subdirectory(src/base) # NOTE, we need to patch the target with version from outside
set_property(TARGET kaldi-base PROPERTY COMPILE_DEFINITIONS "KALDI_VERSION=\"${KALDI_VERSION}\"")
add_subdirectory(src/matrix)
add_subdirectory(src/cudamatrix)
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

add_subdirectory(src/itf)

if(TENSORFLOW_DIR)
    add_subdirectory(src/tfrnnlm)
    add_subdirectory(src/tfrnnlmbin)
endif()

# add all cuda libraries
if(CUDA_FOUND)
    add_subdirectory(src/cudafeat)
    add_subdirectory(src/cudadecoder)
endif()

# add all native executables
add_subdirectory(src/bin)
add_subdirectory(src/gmmbin)
add_subdirectory(src/featbin)
add_subdirectory(src/sgmm2bin)
add_subdirectory(src/fstbin)
add_subdirectory(src/lmbin)
add_subdirectory(src/latbin)
add_subdirectory(src/nnetbin)
add_subdirectory(src/nnet2bin)
add_subdirectory(src/nnet3bin)
add_subdirectory(src/rnnlmbin)
add_subdirectory(src/chainbin)
add_subdirectory(src/ivectorbin)
add_subdirectory(src/onlinebin)
add_subdirectory(src/online2bin)
add_subdirectory(src/kwsbin)

# add all cuda executables
if(CUDA_FOUND)
    add_subdirectory(src/cudafeatbin)
    add_subdirectory(src/cudadecoderbin)
endif()

include(CMakePackageConfigHelpers)
# maybe we should put this into subfolder?
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/kaldi-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/kaldi-config.cmake
    INSTALL_DESTINATION lib/cmake/kaldi
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/kaldi-config-version.cmake
    VERSION ${KALDI_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/kaldi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/cmake/kaldi-config-version.cmake
    DESTINATION lib/cmake/kaldi
)
install(EXPORT kaldi-targets DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/kaldi)
