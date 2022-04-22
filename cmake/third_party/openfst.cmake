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


    if(CONDA_ROOT)

        if(MSVC)
            install(TARGETS fstarcsort fstclosure fstcompile fstcompose fstconcat fstconnect fstconvert fstdeterminize
                            fstdifference fstdisambiguate fstdraw fstencode fstepsnormalize fstequal fstequivalent
                            fstinfo fstintersect fstinvert fstisomorphic fstmap fstminimize fstprint fstproject fstprune
                            fstpush fstrandgen fstrelabel fstreplace fstreverse fstreweight fstrmepsilon fstshortestdistance
                            fstshortestpath fstsymbols fstsynchronize fsttopsort fstunion
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )
            install(TARGETS farcompilestrings farcreate farequal farextract farinfo farisomorphic farprintstrings
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )
            install(TARGETS fstlinear fstloglinearapply
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )
            install(TARGETS mpdtcompose mpdtexpand mpdtinfo mpdtreverse
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )
            install(TARGETS pdtcompose pdtexpand pdtinfo pdtreplace pdtreverse pdtshortestpath
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )
            install(TARGETS fstspecial
            RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT kaldi
            )

            install(DIRECTORY ${openfst_SOURCE_DIR}/src/include/ DESTINATION include/
                    COMPONENT kaldi
                    FILES_MATCHING PATTERN "*.h")

            install(TARGETS fst
                    EXPORT kaldi-targets
                    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT kaldi
                    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT kaldi NAMELINK_SKIP
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT kaldi)
        else()

            install(TARGETS fst
                    LIBRARY
                        DESTINATION ${CMAKE_INSTALL_LIBDIR}
                        COMPONENT kaldi
                        NAMELINK_SKIP # Link to so with version to avoid conflicts with OpenFst 1.8.1 on conda
                    )

        endif()

    else() # Original functionality

        install(DIRECTORY ${openfst_SOURCE_DIR}/src/include/ DESTINATION include/
                FILES_MATCHING PATTERN "*.h")

        install(TARGETS fst
                EXPORT kaldi-targets
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    endif()
endif()
