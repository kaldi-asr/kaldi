all:

include ../kaldi.mk

TESTFILES = lda-estimate-test fmllr-diag-gmm-test

OBJFILES = lda-estimate.o \
    cmvn.o transform-common.o fmllr-diag-gmm.o \
    lvtln.o mllt.o basis-fmllr-diag-gmm.o \
    compressed-transform-stats.o


LIBNAME = kaldi-transform

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../cblasext/kaldi-cblasext.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
