
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet3-xvector-get-egs nnet3-xvector-compute-prob \
           nnet3-xvector-show-progress nnet3-xvector-train \
           nnet3-xvector-compute nnet3-xvector-scoring nnet3-xvector-get-egs-sre \
           nnet3-xvector-get-egs-sre-subsample nnet3-xvector-compute-simple

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../xvector/kaldi-xvector.a ../nnet3/kaldi-nnet3.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a  \
         ../cudamatrix/kaldi-cudamatrix.a \
         ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
         ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
