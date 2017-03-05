

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-pdf-prior.o nnet-randomizer.o

LIBNAME = kaldi-nnet

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk

