
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

OBJFILES = rnnlm-component-itf.o rnnlm-utils.o  rnnlm-nnet.o rnnlm-training.o rnnlm-component.o nnet-parse.o \
           rnnlm-diagnostics.o

LIBNAME = kaldi-rnnlm

ADDLIBS = ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
