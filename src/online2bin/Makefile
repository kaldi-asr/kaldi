
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = online2-wav-gmm-latgen-faster apply-cmvn-online \
     extend-wav-with-silence compress-uncompress-speex \
     ivector-extract-online2 \
     online2-wav-dump-features ivector-randomize \
     online2-wav-nnet3-latgen-faster online2-wav-nnet3-latgen-grammar \
     online2-tcp-nnet3-decode-faster

OBJFILES =

TESTFILES =

ADDLIBS = ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
          ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../cblasext/kaldi-cblasext.a ../base/kaldi-base.a 
include ../makefiles/default_rules.mk
