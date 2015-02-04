
all:

include ../kaldi.mk

TESTFILES =

OBJFILES = online-gmm-decodable.o online-feature-pipeline.o online-ivector-feature.o \
           online-nnet2-feature-pipeline.o online-gmm-decoding.o online-timing.o \
           online-endpoint.o onlinebin-util.o online-speex-wrapper.o \
           online-nnet2-decoding.o online-nnet2-decoding-threaded.o

LIBNAME = kaldi-online2

ADDLIBS = ../gmm/kaldi-gmm.a ../transform/kaldi-transform.a ../feat/kaldi-feat.a \
     ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a \
     ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a ../hmm/kaldi-hmm.a \
     ../thread/kaldi-thread.a ../ivector/kaldi-ivector.a \
     ../cudamatrix/kaldi-cudamatrix.a ../nnet2/kaldi-nnet2.a


include ../makefiles/default_rules.mk

