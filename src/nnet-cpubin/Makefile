
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

BINFILES = nnet-randomize-frames nnet-am-info nnet-train nnet-init \
   nnet-train-simple nnet-train-transitions nnet-latgen-faster nnet-am-copy \
   nnet-am-init nnet-insert nnet-align-compiled \
   nnet-compute-prob nnet-copy-egs nnet-shrink nnet-combine nnet-combine-a \
   nnet-am-average nnet-am-combine nnet-am-compute nnet-am-shrink nnet-am-mixup \
   nnet-train-lbfgs nnet-get-egs nnet-train-parallel nnet-gradient \
   nnet-get-preconditioner nnet-precondition nnet-select-egs nnet-combine-fast \
   nnet-subset-egs nnet-shuffle-egs nnet-am-fix nnet-logprob  nnet-logprob2 \
   nnet-logprob2-parallel nnet-logprob-parallel nnet-am-stats nnet-am-rescale \
   nnet-am-limit-rank nnet-latgen-faster-parallel

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet-cpu/kaldi-nnet-cpu.a ../gmm/kaldi-gmm.a \
          ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
