
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = ivector-extractor-init ivector-extractor-acc-stats \
           ivector-extractor-sum-accs ivector-extractor-est \
           ivector-extract compute-vad select-voiced-frames \
           compute-vad-from-frame-likes merge-vads \
           ivector-normalize-length \
           ivector-transform ivector-compute-dot-products ivector-mean \
           ivector-compute-lda ivector-compute-plda \
	   ivector-copy-plda compute-eer \
           ivector-subtract-global-mean ivector-plda-scoring \
           logistic-regression-train logistic-regression-eval \
           logistic-regression-copy create-split-from-vad \
           ivector-extract-online ivector-adapt-plda

OBJFILES =



TESTFILES =


ADDLIBS = ../ivector/kaldi-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
