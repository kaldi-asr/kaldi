
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare

TESTFILES = language-model-test

OBJFILES = language-model.o # ctc-functions.o

LIBNAME = kaldi-ctc

ADDLIBS = ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../cudamatrix/kaldi-cudamatrix.a ../util/kaldi-util.a \
          ../base/kaldi-base.a


include ../makefiles/default_rules.mk

