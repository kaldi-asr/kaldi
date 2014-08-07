
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-best-path lattice-prune lattice-equivalent lattice-to-nbest \
           lattice-lmrescore lattice-scale lattice-union lattice-to-post \
           lattice-determinize lattice-oracle lattice-rmali \
           lattice-compose lattice-boost-ali lattice-copy lattice-to-fst \
           lattice-to-phone-lattice lattice-interp lattice-project \
           lattice-add-trans-probs lattice-difference lattice-word-align \
           nbest-to-linear nbest-to-lattice lattice-1best linear-to-nbest \
           lattice-mbr-decode lattice-align-words lattice-to-mpe-post \
           lattice-copy-backoff nbest-to-ctm lattice-determinize-pruned \
           lattice-to-ctm-conf lattice-arcgraph lattice-combine lattice-reverse \
           lattice-rescore-mapped lattice-depth lattice-align-phones \
           lattice-to-smbr-post lattice-determinize-pruned-parallel \
           lattice-add-penalty lattice-align-words-lexicon lattice-push \
           lattice-minimize lattice-limit-depth lattice-depth-per-frame \
           lattice-determinize-phone-pruned lattice-determinize-phone-pruned-parallel \
           lattice-expand-ngram


OBJFILES =



TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../thread/kaldi-thread.a \
					../base/kaldi-base.a 

include ../makefiles/default_rules.mk

