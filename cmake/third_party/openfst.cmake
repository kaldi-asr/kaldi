cmake_minimum_required(VERSION 3.14)
include(FetchContent)

FetchContent_Declare(
        openfst
        GIT_REPOSITORY  https://github.com/kkm000/openfst
        GIT_TAG         338225416178ac36b8002d70387f5556e44c8d05 # tag win/1.7.2.1
)

FetchContent_GetProperties(openfst)
if(NOT openfst_POPULATED)
    FetchContent_Populate(openfst)
    include_directories(${openfst_SOURCE_DIR}/src/include)

    add_subdirectory(${openfst_SOURCE_DIR} ${openfst_BINARY_DIR})

    install(DIRECTORY ${openfst_SOURCE_DIR}/src/include/ DESTINATION include/
            FILES_MATCHING PATTERN "*.h")

    install(TARGETS fst
            EXPORT kaldi-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
