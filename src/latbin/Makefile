
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = lattice-best-path lattice-prune lattice-equivalent lattice-to-nbest \
           lattice-lmrescore lattice-scale lattice-union lattice-to-post \
           lattice-determinize lattice-oracle lattice-rmali \
           lattice-compose lattice-boost-ali lattice-copy lattice-to-fst \
           lattice-to-phone-lattice lattice-interp lattice-project \
           lattice-add-trans-probs lattice-difference \
           nbest-to-linear nbest-to-lattice lattice-1best linear-to-nbest \
           lattice-mbr-decode lattice-align-words lattice-to-mpe-post \
           lattice-copy-backoff nbest-to-ctm lattice-determinize-pruned \
           lattice-to-ctm-conf lattice-combine \
           lattice-rescore-mapped lattice-depth lattice-align-phones \
           lattice-to-smbr-post lattice-determinize-pruned-parallel \
           lattice-add-penalty lattice-align-words-lexicon lattice-push \
           lattice-minimize lattice-limit-depth lattice-depth-per-frame \
           lattice-confidence lattice-determinize-phone-pruned \
           lattice-determinize-phone-pruned-parallel lattice-expand-ngram \
           lattice-lmrescore-const-arpa lattice-lmrescore-rnnlm nbest-to-prons \
           lattice-arc-post lattice-determinize-non-compact lattice-lmrescore-kaldi-rnnlm \
           lattice-lmrescore-pruned lattice-lmrescore-kaldi-rnnlm-pruned lattice-reverse

OBJFILES =


TESTFILES =

ADDLIBS = ../rnnlm/kaldi-rnnlm.a ../nnet3/kaldi-nnet3.a \
          ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a ../lm/kaldi-lm.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
