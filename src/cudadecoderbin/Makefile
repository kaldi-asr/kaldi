all:

include ../kaldi.mk

ifeq ($(CUDA), true)

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = batched-wav-nnet3-cuda

OBJFILES =

TESTFILES =

ADDLIBS = ../cudadecoder/kaldi-cudadecoder.a  ../cudafeat/kaldi-cudafeat.a \
../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
../feat/kaldi-feat.a ../transform/kaldi-transform.a \
../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
../matrix/kaldi-matrix.a ../base/kaldi-base.a

endif

include ../makefiles/default_rules.mk
