# Try to find the OpenFst library and headers.
#  OpenFst_ROOT_DIR     - where to find

#  OpenFst_FOUND        - system has OpenFst
#  OpenFst_INCLUDE_DIRS - the OpenFst include directory
#  OpenFst_LIBRARIES    - the OpenFst libraries


find_path(OpenFst_INCLUDE_DIRS
    NAMES fst/fst.h
    HINTS
        ${OPENFST_ROOT_DIR}
        ${OPENFST_ROOT_DIR}/src/include
    DOC "The directory where OpenFst includes reside"
)

find_library(OpenFst_LIBRARIES
    NAMES fst
    HINTS "${OPENFST_ROOT_DIR}/lib" "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/openfst"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenFst
    FOUND_VAR OpenFst_FOUND
    REQUIRED_VARS OpenFst_INCLUDE_DIRS OpenFst_LIBRARIES
)

mark_as_advanced(OpenFst_FOUND)

add_library(OpenFst INTERFACE)
target_include_directories(OpenFst INTERFACE ${CUB_INCLUDE_DIR})
