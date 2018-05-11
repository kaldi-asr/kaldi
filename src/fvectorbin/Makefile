
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = fvector-add-noise fvector-chunk fvector-get-egs \
		   fvector-chunk-block fvector-add-noise-block fvector-get-egs-block \
		   compute-wav-to-rawmatrix fvector-debug-write-to-wav \
		   fvector-debug-wav-to-vector fvector-debug-check-filter-bank \
		   fvector-chunk-separate fvector-add-noise-separate

OBJFILES =



TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../fvector/kaldi-fvector.a \
		  ../feat/kaldi-feat.a ../base/kaldi-base.a ../nnet3/kaldi-nnet3.a \
		  ../cudamatrix/kaldi-cudamatrix.a

include ../makefiles/default_rules.mk
