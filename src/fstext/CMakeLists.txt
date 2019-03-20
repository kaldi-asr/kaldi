set(source_files push-special.cc kaldi-fst-io.cc context-fst.cc
  grammar-context-fst.cc)

set(test_files determinize-star-test
  pre-determinize-test trivial-factor-weight-test
  context-fst-test factor-test table-matcher-test fstext-utils-test
  remove-eps-local-test lattice-weight-test
  determinize-lattice-test lattice-utils-test deterministic-fst-test
  push-special-test epsilon-property-test prune-special-test)

set(dependencies base util matrix tree)

create_library(fstext "${source_files}" "${test_files}" "${dependencies}")
