
all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =
include ../kaldi.mk

TESTFILES = fvector-perturb-test

OBJFILES = fvector-perturb.o

LIBNAME = kaldi-fvector

ADDLIBS = ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../feat/kaldi-feat.a


include ../makefiles/default_rules.mk
