set(PUBLIC_HEADERS
    kaldi-kws.h
    kws-functions.h
    kws-scoring.h
)

add_library(kaldi-kws
    kws-functions.cc
    kws-functions2.cc
    kws-scoring.cc
)

target_include_directories(kaldi-kws PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-kws PUBLIC
    kaldi-lat
    kaldi-hmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)


install(TARGETS kaldi-kws
    EXPORT kaldi-kws-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/kws)

