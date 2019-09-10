set(PUBLIC_HEADERS
    simple-decoder.h
    decodable-sum.h
    biglm-faster-decoder.h
    lattice-faster-online-decoder.h
    faster-decoder.h
    lattice-faster-decoder.h
    training-graph-compiler.h
    lattice-simple-decoder.h
    lattice-biglm-faster-decoder.h
    decodable-mapped.h
    decoder-wrappers.h
    decodable-matrix.h
    grammar-fst.h
)

add_library(kaldi-decoder
    decoder-wrappers.cc
    lattice-simple-decoder.cc
    decodable-matrix.cc
    grammar-fst.cc
    lattice-faster-decoder.cc
    training-graph-compiler.cc
    simple-decoder.cc
    lattice-faster-online-decoder.cc
    faster-decoder.cc
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

