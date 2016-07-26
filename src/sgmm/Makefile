all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =
include ../kaldi.mk

TESTFILES = am-sgmm-test estimate-am-sgmm-test fmllr-sgmm-test \
	    estimate-am-sgmm-multi-test

OBJFILES = am-sgmm.o estimate-am-sgmm.o fmllr-sgmm.o sgmm-clusterable.o \
           estimate-am-sgmm-ebw.o estimate-am-sgmm-multi.o decodable-am-sgmm.o

LIBNAME = kaldi-sgmm
ADDLIBS = ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
