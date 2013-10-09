

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o nnet-cache.o \
           nnet-cache-tgtmat.o nnet-loss-prior.o nnet-pdf-prior.o

LIBNAME = kaldi-nnet

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

