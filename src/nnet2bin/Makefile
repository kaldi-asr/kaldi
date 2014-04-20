
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet-randomize-frames nnet-am-info nnet-init \
   nnet-train-simple nnet-train-ensemble nnet-train-transitions nnet-latgen-faster nnet-am-copy \
   nnet-am-init nnet-insert nnet-align-compiled \
   nnet-compute-prob nnet-copy-egs nnet-shrink nnet-combine nnet-combine-a \
   nnet-am-average nnet-am-combine nnet-am-compute nnet-am-shrink nnet-am-mixup \
   nnet-train-lbfgs nnet-get-egs nnet-train-parallel nnet-gradient \
   nnet-get-preconditioner nnet-precondition nnet-select-egs nnet-combine-fast \
   nnet-subset-egs nnet-shuffle-egs nnet-am-fix nnet-logprob  nnet-logprob2 \
   nnet-logprob2-parallel nnet-logprob-parallel nnet-am-stats nnet-am-rescale \
   nnet-am-limit-rank nnet-latgen-faster-parallel nnet-to-raw-nnet nnet-compute \
   raw-nnet-concat nnet-am-limit-rank-final raw-nnet-info \
   nnet-get-feature-transform nnet-compute-from-egs \
   nnet-am-widen nnet-show-progress nnet-limit-degradation \
   nnet-get-feature-transform-multi nnet-copy-egs-discriminative \
   nnet-get-egs-discriminative nnet-shuffle-egs-discriminative \
   nnet-compare-hash-discriminative nnet-combine-egs-discriminative \
   nnet-train-discriminative-simple nnet-train-discriminative-parallel \
   nnet-modify-learning-rates nnet-normalize-stddev nnet-perturb-egs \
   nnet-perturb-egs-fmllr nnet-get-weighted-egs nnet-adjust-priors \
   cuda-compiled nnet-replace-last-layers

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet2/kaldi-nnet2.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
         ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
