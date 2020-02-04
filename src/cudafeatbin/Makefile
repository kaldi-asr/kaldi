src/cudafeatbin/Makefile
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
ifeq ($(CUDA), true)
ifeq ($(WITH_CUDAFEAT), 1)

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES =

BINFILES += compute-mfcc-feats-cuda \
						apply-cmvn-online-cuda \
						compute-online-feats-cuda \
						compute-fbank-feats-cuda \
						apply-batched-cmvn-online-cuda \
						compute-mfcc-online-batched-cuda \
						compute-online-feats-batched-cuda

OBJFILES =

TESTFILES =

ADDLIBS = ../cudafeat/kaldi-cudafeat.a ../online2/kaldi-online2.a  \
          ../ivector/kaldi-ivector.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../cudamatrix/kaldi-cudamatrix.a \
          ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

else
	$(warning "Not building cudafeat extension -- please configure with --with-cudafeat[=true]")
endif

endif
include ../makefiles/default_rules.mk
