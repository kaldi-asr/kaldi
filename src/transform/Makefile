all:

include ../kaldi.mk

TESTFILES = regtree-fmllr-diag-gmm-test lda-estimate-test \
      regression-tree-test fmllr-diag-gmm-test \
      regtree-mllr-diag-gmm-test fmpe-test fmllr-raw-test

OBJFILES = regression-tree.o regtree-mllr-diag-gmm.o lda-estimate.o \
    regtree-fmllr-diag-gmm.o cmvn.o transform-common.o fmllr-diag-gmm.o \
    lvtln.o mllt.o hlda.o fmpe.o basis-fmllr-diag-gmm.o \
    compressed-transform-stats.o fmllr-raw.o decodable-am-diag-gmm-regtree.o


LIBNAME = kaldi-transform

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
