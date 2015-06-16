
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


TESTFILES = natural-gradient-online-test nnet-graph-test \
						nnet-descriptor-test nnet-parse-test

OBJFILES = nnet-compile.o nnet-component-itf.o nnet-simple-component.o \
  nnet-general-component.o nnet-parse.o natural-gradient-online.o \
  nnet-descriptor.o nnet-optimize.o nnet-computation.o \
  nnet-computation-graph.o nnet-graph.o am-nnet-simple.o \
  nnet-example.o nnet-compile-utils.o

LIBNAME = kaldi-nnet3

ADDLIBS = ../thread/kaldi-thread.a ../lat/kaldi-lat.a ../gmm/kaldi-gmm.a \
      ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
      ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
      ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk
