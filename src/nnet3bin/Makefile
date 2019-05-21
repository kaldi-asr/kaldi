
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet3-init nnet3-info nnet3-get-egs nnet3-copy-egs nnet3-subset-egs \
   nnet3-shuffle-egs nnet3-acc-lda-stats nnet3-merge-egs \
   nnet3-compute-from-egs nnet3-train nnet3-am-init nnet3-am-train-transitions \
   nnet3-am-adjust-priors nnet3-am-copy nnet3-compute-prob \
   nnet3-average nnet3-am-info nnet3-combine nnet3-latgen-faster \
   nnet3-latgen-faster-parallel nnet3-show-progress nnet3-align-compiled \
   nnet3-copy nnet3-get-egs-dense-targets nnet3-compute \
   nnet3-discriminative-get-egs nnet3-discriminative-copy-egs \
   nnet3-discriminative-merge-egs nnet3-discriminative-shuffle-egs \
   nnet3-discriminative-compute-objf nnet3-discriminative-train \
   nnet3-discriminative-subset-egs nnet3-get-egs-simple \
   nnet3-discriminative-compute-from-egs nnet3-latgen-faster-looped \
   nnet3-egs-augment-image nnet3-xvector-get-egs nnet3-xvector-compute \
   nnet3-latgen-grammar nnet3-compute-batch nnet3-latgen-faster-batch

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
