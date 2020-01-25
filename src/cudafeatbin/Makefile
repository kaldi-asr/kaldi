
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES =

ifeq ($(CUDA), true)
  BINFILES += compute-mfcc-feats-cuda apply-cmvn-online-cuda compute-online-feats-cuda compute-fbank-feats-cuda
endif

OBJFILES =

TESTFILES =

ADDLIBS = ../cudafeat/kaldi-cudafeat.a ../online2/kaldi-online2.a  \
          ../ivector/kaldi-ivector.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../cudamatrix/kaldi-cudamatrix.a \
          ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
