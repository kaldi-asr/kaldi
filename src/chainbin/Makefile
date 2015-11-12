
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = chain-est-phone-lm


OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
         ../thread/kaldi-thread.a ../cudamatrix/kaldi-cudamatrix.a \
         ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
         ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
