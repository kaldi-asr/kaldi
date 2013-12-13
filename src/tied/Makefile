all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = am-tied-diag-gmm-test am-tied-full-gmm-test

OBJFILES = tied-gmm.o mle-tied-gmm.o am-tied-diag-gmm.o mle-am-tied-diag-gmm.o \
    am-tied-full-gmm.o mle-am-tied-full-gmm.o decodable-am-tied-diag-gmm.o \
    decodable-am-tied-full-gmm.o

LIBNAME = kaldi-tied

ADDLIBS = ../base/kaldi-base.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
	../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

