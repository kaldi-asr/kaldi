cmake_minimum_required(VERSION 2.8.2)
project(openfst-download NONE)

include(ExternalProject)
ExternalProject_Add(openfst
    GIT_REPOSITORY https://github.com/kkm000/openfst
    GIT_TAG winport
    SOURCE_DIR "${CMAKE_BINARY_DIR}/openfst"
    BINARY_DIR ""
    PATCH_COMMAND "git" "apply" "--cached" "${PROJECT_SOURCE_DIR}/cmake/third_party/openfst.patch"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
