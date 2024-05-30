
all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =
include ../kaldi.mk

TESTFILES = ivector-extractor-test plda-test logistic-regression-test

OBJFILES = ivector-extractor.o voice-activity-detection.o plda.o \
           logistic-regression.o agglomerative-clustering.o

LIBNAME = kaldi-ivector

ADDLIBS = ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 


include ../makefiles/default_rules.mk
