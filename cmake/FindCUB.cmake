# Try to find the CUB library and headers.
#  CUB_ROOT_DIR     - where to find

#  CUB_FOUND        - system has CUB
#  CUB_INCLUDE_DIRS - the CUB include directory


find_path(CUB_INCLUDE_DIR
    NAMES cub/cub.cuh
    HINTS ${CUB_ROOT_DIR}
    DOC "The directory where CUB includes reside"
)

set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB
        FOUND_VAR CUB_FOUND
        REQUIRED_VARS CUB_INCLUDE_DIR
)

mark_as_advanced(CUB_FOUND)

add_library(CUB INTERFACE)
target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
