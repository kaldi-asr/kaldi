set(source_files kaldi-lattice.cc lattice-functions.cc word-align-lattice.cc
  phone-align-lattice.cc word-align-lattice-lexicon.cc sausages.cc
  push-lattice.cc minimize-lattice.cc determinize-lattice-pruned.cc
  confidence.cc compose-lattice-pruned.cc)

set(test_files kaldi-lattice-test push-lattice-test minimize-lattice-test
  determinize-lattice-pruned-test word-align-lattice-lexicon-test)
set(dependencies base util hmm tree matrix)
create_library(lat "${source_files}" "${test_files}" "${dependencies}")
