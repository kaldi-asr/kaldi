set(PUBLIC_HEADERS
    biglm-faster-decoder.h
    decodable-mapped.h
    decodable-matrix.h
    decodable-sum.h
    decoder-wrappers.h
    faster-decoder.h
    grammar-fst.h
    lattice-biglm-faster-decoder.h
    lattice-faster-decoder.h
    lattice-faster-online-decoder.h
    lattice-simple-decoder.h
    simple-decoder.h
    training-graph-compiler.h
)

add_library(kaldi-decoder
    decodable-matrix.cc
    decoder-wrappers.cc
    faster-decoder.cc
    grammar-fst.cc
    lattice-faster-decoder.cc
    lattice-faster-online-decoder.cc
    lattice-simple-decoder.cc
    simple-decoder.cc
    training-graph-compiler.cc
)

target_include_directories(kaldi-decoder PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-decoder PUBLIC
    kaldi-lat
    kaldi-fstext
    kaldi-hmm
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-base
)


install(TARGETS kaldi-decoder
    EXPORT kaldi-decoder-targets
    INCLUDES DESTINATION include/kaldi
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/decoder)

