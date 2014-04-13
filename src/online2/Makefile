
all:

include ../kaldi.mk

TESTFILES = online-feature-test

OBJFILES = online-feature.o online-gmm-decodable.o online-feature-pipeline.o \
           online-gmm-decoding.o online-timing.o online-endpoint.o onlinebin-util.o \
           online-speex-wrapper.o

LIBNAME = kaldi-online2

ADDLIBS = ../gmm/kaldi-gmm.a ../transform/kaldi-transform.a ../feat/kaldi-feat.a \
     ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a \
     ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a ../hmm/kaldi-hmm.a


include ../makefiles/default_rules.mk

