
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = online2-wav-gmm-latgen-faster apply-cmvn-online \
     extend-wav-with-silence compress-uncompress-speex \
     online2-wav-nnet2-latgen-faster ivector-extract-online2 \
     online2-wav-dump-features ivector-randomize \
     online2-wav-nnet2-am-compute

OBJFILES = 

TESTFILES =

ADDLIBS = ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
           ../nnet2/kaldi-nnet2.a ../lat/kaldi-lat.a \
          ../decoder/kaldi-decoder.a  ../cudamatrix/kaldi-cudamatrix.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../thread/kaldi-thread.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
