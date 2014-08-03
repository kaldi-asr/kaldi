

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


TESTFILES = nnet-component-test nnet-precondition-test \
	nnet-precondition-online-test nnet-example-functions-test \
    nnet-nnet-test am-nnet-test online-nnet2-decodable-test

OBJFILES = nnet-component.o nnet-nnet.o train-nnet.o train-nnet-ensemble.o nnet-update.o \
     nnet-randomize.o nnet-compute.o am-nnet.o nnet-functions.o  \
     nnet-precondition.o shrink-nnet.o combine-nnet.o combine-nnet-a.o \
     mixup-nnet.o nnet-lbfgs.o nnet-update-parallel.o combine-nnet-fast.o \
     nnet-fix.o nnet-stats.o rescale-nnet.o nnet-limit-rank.o nnet-example.o \
     get-feature-transform.o widen-nnet.o nnet-precondition-online.o \
     nnet-example-functions.o nnet-compute-discriminative.o \
     nnet-compute-discriminative-parallel.o online-nnet2-decodable.o \
     train-nnet-perturbed.o

LIBNAME = kaldi-nnet2

ADDLIBS = ../thread/kaldi-thread.a ../lat/kaldi-lat.a ../gmm/kaldi-gmm.a \
      ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
      ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
      ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk
