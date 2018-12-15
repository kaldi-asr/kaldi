all:

# This directory contains code related to the adaptation
# framework in ../adapt, for nnet3 and (principally) chain
# training.

include ../kaldi.mk

TESTFILES =

OBJFILES =

LIBNAME = kaldi-nnet3a

ADDLIBS = ../fstext/kaldi-fstext.a ../chain/kaldi-chain.a \
          ../nnet3/kaldi-nnet3.a  ../adapt/kaldi-adapt.a \
           ../cudamatrix/kaldi-cudamatrix.a ../hmm/kaldi-hmm.a	\
           ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
           ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../lat/kaldi-lat.a \
           ../matrix-kaldi-matrix.a ../util/kaldi-util.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
