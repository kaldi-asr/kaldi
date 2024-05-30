
# cuda-compiled was moved to nnet3bin/; remove it in case an outdated
# version persists.
all:
	-rm -f cuda-compiled


EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet-am-info nnet-init \
   nnet-train-simple nnet-train-ensemble nnet-train-transitions nnet-latgen-faster nnet-am-copy \
   nnet-am-init nnet-insert nnet-align-compiled \
   nnet-compute-prob nnet-copy-egs nnet-combine \
   nnet-am-average nnet-am-compute nnet-am-mixup \
   nnet-get-egs nnet-train-parallel nnet-combine-fast \
   nnet-subset-egs nnet-shuffle-egs nnet-am-fix \
   nnet-latgen-faster-parallel nnet-to-raw-nnet nnet-compute \
   raw-nnet-concat raw-nnet-info \
   nnet-get-feature-transform nnet-compute-from-egs \
   nnet-am-widen nnet-show-progress \
   nnet-get-feature-transform-multi nnet-copy-egs-discriminative \
   nnet-get-egs-discriminative nnet-shuffle-egs-discriminative \
   nnet-compare-hash-discriminative nnet-combine-egs-discriminative \
   nnet-train-discriminative-simple nnet-train-discriminative-parallel \
   nnet-modify-learning-rates nnet-normalize-stddev  \
   nnet-get-weighted-egs nnet-adjust-priors \
   nnet-replace-last-layers nnet-am-switch-preconditioning \
   nnet1-to-raw-nnet raw-nnet-copy nnet-relabel-egs nnet-am-reinitialize

OBJFILES =


TESTFILES =

ADDLIBS = ../nnet2/kaldi-nnet2.a ../nnet/kaldi-nnet.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
