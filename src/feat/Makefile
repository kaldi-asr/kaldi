

all:

include ../kaldi.mk

TESTFILES = feature-mfcc-test feature-fbank-test \
         feature-functions-test pitch-functions-test feature-sdc-test \
         resample-test online-feature-test signal-test wave-reader-test

OBJFILES = feature-functions.o feature-mfcc.o feature-fbank.o \
           mel-computations.o wave-reader.o \
           pitch-functions.o resample.o online-feature.o signal.o \
           feature-window.o

LIBNAME = kaldi-feat

ADDLIBS = ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../cblasext/kaldi-cblasext.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
