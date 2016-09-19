
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


TESTFILES = natural-gradient-online-test nnet-graph-test \
  nnet-descriptor-test nnet-parse-test nnet-component-test \
  nnet-compile-utils-test nnet-nnet-test nnet-utils-test \
  nnet-compile-test nnet-analyze-test nnet-compute-test \
  nnet-optimize-test nnet-derivative-test nnet-example-test \
  nnet-common-test

OBJFILES = nnet-common.o nnet-compile.o nnet-component-itf.o \
  nnet-simple-component.o \
  nnet-general-component.o nnet-parse.o natural-gradient-online.o \
  nnet-descriptor.o nnet-optimize.o nnet-computation.o \
  nnet-computation-graph.o nnet-graph.o am-nnet-simple.o \
  nnet-example.o nnet-nnet.o nnet-compile-utils.o \
  nnet-utils.o nnet-compute.o nnet-test-utils.o nnet-analyze.o \
  nnet-example-utils.o nnet-training.o \
  nnet-diagnostics.o nnet-combine.o nnet-am-decodable-simple.o \
  nnet-optimize-utils.o nnet-chain-example.o \
  nnet-chain-training.o nnet-chain-diagnostics.o nnet-chain-combine.o \
  discriminative-supervision.o nnet-discriminative-example.o \
  nnet-discriminative-diagnostics.o \
  discriminative-training.o nnet-discriminative-training.o \
  online-nnet3-decodable-simple.o


LIBNAME = kaldi-nnet3

ADDLIBS = ../chain/kaldi-chain.a ../cudamatrix/kaldi-cudamatrix.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
