cmake_minimum_required(VERSION 2.8.2)
project(openfst-download NONE)

include(ExternalProject)
ExternalProject_Add(openfst
    GIT_REPOSITORY https://github.com/kkm000/openfst
    GIT_TAG 0bca6e76d24647427356dc242b0adbf3b5f1a8d9 # tag win/1.7.2.1
    SOURCE_DIR "${CMAKE_BINARY_DIR}/openfst"
    BINARY_DIR ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
