
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = rnnlm-get-egs rnnlm-train rnnlm-get-sampling-lm \
           rnnlm-get-word-embedding rnnlm-compute-prob rnnlm-sentence-probs

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../rnnlm/kaldi-rnnlm.a ../nnet3/kaldi-nnet3.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
