cmake_minimum_required(VERSION 2.8.2)
project(cub-download NONE)

include(ExternalProject)
ExternalProject_Add(cub
    GIT_REPOSITORY https://github.com/NVlabs/cub
    GIT_TAG c3cceac115c072fb63df1836ff46d8c60d9eb304 # tag v1.8.0
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub"
    BINARY_DIR ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)
