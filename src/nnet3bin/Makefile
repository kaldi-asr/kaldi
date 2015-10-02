
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
   nnet3-copy nnet3-show-progress nnet3-align-compiled \
   nnet3-get-egs-dense-targets nnet3-compute

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../nnet3/kaldi-nnet3.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
         ../thread/kaldi-thread.a ../cudamatrix/kaldi-cudamatrix.a \
         ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
         ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
