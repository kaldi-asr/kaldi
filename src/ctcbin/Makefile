
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = ctc-init-transition-model ctc-copy-transition-model \
    ctc-get-supervision \
	nnet3-ctc-get-egs nnet3-ctc-copy-egs nnet3-ctc-subset-egs \
    nnet3-ctc-shuffle-egs nnet3-ctc-merge-egs nnet3-ctc-train \
    nnet3-ctc-copy nnet3-ctc-info nnet3-ctc-compute-prob

# nnet3-ctc-{copy,subset,shuffle,merge}-egs should all be fairly
# simple copy-and-modify from the corresponding nnet3-*-egs programs.


OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../nnet3/kaldi-nnet3.a ../ctc/kaldi-ctc.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
         ../thread/kaldi-thread.a ../cudamatrix/kaldi-cudamatrix.a \
         ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
         ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
