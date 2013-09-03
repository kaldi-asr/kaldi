

all:

include ../kaldi.mk

TESTFILES = nnet-component-test nnet-precondition-test

OBJFILES = nnet-component.o nnet-nnet.o nnet-update.o train-nnet.o \
     nnet-randomize.o nnet-compute.o am-nnet.o nnet-functions.o  \
     nnet-precondition.o shrink-nnet.o combine-nnet.o combine-nnet-a.o \
     mixup-nnet.o nnet-lbfgs.o nnet-update-parallel.o combine-nnet-fast.o \
     nnet-fix.o nnet-stats.o rescale-nnet.o nnet-limit-rank.o nnet-example.o \
     get-feature-transform.o widen-nnet.o

LIBNAME = kaldi-nnet2

ADDLIBS = ../thread/kaldi-thread.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a \
     ../transform/kaldi-transform.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
     ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk
