all:

include ../kaldi.mk

TESTFILES = differentiable-fmllr-test

OBJFILES = differentiable-fmllr.o

LIBNAME = kaldi-adapt

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../hmm/kaldi-hmm.a	\
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
