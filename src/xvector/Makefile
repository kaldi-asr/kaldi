
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = xvector-test

OBJFILES = xvector.o nnet-xvector-training.o nnet-xvector-diagnostics.o nnet-xvector-compute.o

LIBNAME = kaldi-xvector

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
          ../nnet3/kaldi-nnet3.a ../cudamatrix/kaldi-cudamatrix.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
          ../util/kaldi-util.a

include ../makefiles/default_rules.mk
