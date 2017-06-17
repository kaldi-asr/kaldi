all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =
include ../kaldi.mk

TESTFILES = am-sgmm2-test estimate-am-sgmm2-test  \
   fmllr-sgmm2-test

OBJFILES = am-sgmm2.o estimate-am-sgmm2.o estimate-am-sgmm2-ebw.o fmllr-sgmm2.o \
          am-sgmm2-project.o decodable-am-sgmm2.o

LIBNAME = kaldi-sgmm2

ADDLIBS = ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
