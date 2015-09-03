
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = ivector-extractor-test plda-test logistic-regression-test

OBJFILES = ivector-extractor.o voice-activity-detection.o plda.o logistic-regression.o

LIBNAME = kaldi-ivector

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
		../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
        ../util/kaldi-util.a 

include ../makefiles/default_rules.mk
