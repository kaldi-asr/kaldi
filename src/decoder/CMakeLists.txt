set(source_files training-graph-compiler.cc lattice-simple-decoder.cc
  lattice-faster-decoder.cc lattice-faster-online-decoder.cc simple-decoder.cc
  faster-decoder.cc  decoder-wrappers.cc grammar-fst.cc decodable-matrix.cc)
set(test_files "")
set(dependencies base util matrix gmm hmm tree transform lat)
create_library(decoder "${source_files}" "${test_files}" "${dependencies}")
