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

