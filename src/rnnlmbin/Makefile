
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = rnnlm-get-egs rnnlm-eval rnnlm-train rnnlm-init rnnlm-copy rnnlm-info \
           rnnlm-tmp \
           rnnlm-compute-prob rnnlm-show-progress

OBJFILES =

cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../rnnlm/kaldi-rnnlm.a ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

#ADDLIBS = ../util/kaldi-util.a ../thread/kaldi-thread.a \
#          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
#          ../nnet3/kaldi-nnet3.a ../hmm/kaldi-hmm.a ../cudamatrix/kaldi-cudamatrix.a

include ../makefiles/default_rules.mk
