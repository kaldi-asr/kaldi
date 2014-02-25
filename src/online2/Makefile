
all:

include ../kaldi.mk

TESTFILES = online-feature-test

OBJFILES = online-feature.o online-gmm-decodable.o online-feature-pipeline.o \
           online-gmm-decoding.o online-timing.o onlinebin-util.o

LIBNAME = kaldi-online2

ADDLIBS = ../gmm/kaldi-gmm.a ../transform/kaldi-transform.a ../feat/kaldi-feat.a \
     ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

