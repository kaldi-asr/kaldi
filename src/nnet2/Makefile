

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


TESTFILES = nnet-component-test nnet-precondition-test \
	nnet-precondition-online-test nnet-example-functions-test \
    nnet-nnet-test am-nnet-test online-nnet2-decodable-test \
    nnet-compute-test

OBJFILES = nnet-component.o nnet-nnet.o train-nnet.o train-nnet-ensemble.o nnet-update.o \
     nnet-compute.o am-nnet.o nnet-functions.o  \
     nnet-precondition.o combine-nnet.o \
     mixup-nnet.o nnet-update-parallel.o combine-nnet-fast.o \
     nnet-fix.o nnet-stats.o rescale-nnet.o nnet-limit-rank.o nnet-example.o \
     get-feature-transform.o widen-nnet.o nnet-precondition-online.o \
     nnet-example-functions.o nnet-compute-discriminative.o \
     nnet-compute-discriminative-parallel.o online-nnet2-decodable.o \
     nnet-compute-online.o

LIBNAME = kaldi-nnet2

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
