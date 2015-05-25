
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


TESTFILES = natural-gradient-online-test 

OBJFILES = nnet-compile.o nnet-component.o nnet-parse.o natural-gradient-online.o \
  nnet-component-itf.o nnet-optimize.o nnet-computation.o \
  nnet-computation-graph.o nnet-graph.o nnet-utils.o am-nnet-simple.o \
  nnet-compute.o nnet-example.o nnet-common.o

LIBNAME = kaldi-nnet3

ADDLIBS = ../thread/kaldi-thread.a ../lat/kaldi-lat.a ../gmm/kaldi-gmm.a \
      ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
      ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
      ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk
