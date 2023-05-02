all: ;

include ../kaldi.mk

ifeq ($(CUDA), true)
ifeq ($(WITH_CUDADECODER), true)

# Make sure we have CUDA_ARCH from kaldi.mk,
ifndef CUDA_ARCH
  $(error CUDA_ARCH is undefined, run 'src/configure')
endif

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
EXTRA_LDLIBS += -lrt

BINFILES = batched-wav-nnet3-cuda2 batched-wav-nnet3-cuda-online

# The legacy batched-wav-nnet3-cuda contains some cuda calls not available on Jetson devices
ifeq ($(HOST_ARCH), x86_64)
  BINFILES += batched-wav-nnet3-cuda
endif

OBJFILES =

TESTFILES =

ADDLIBS = ../cudadecoder/kaldi-cudadecoder.a  ../cudafeat/kaldi-cudafeat.a \
			../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
			../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
			../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
			../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
			../feat/kaldi-feat.a ../transform/kaldi-transform.a \
			../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
			../matrix/kaldi-matrix.a ../base/kaldi-base.a

else
all:
	$(warning "Not building cudadecoder extension -- to build with it, configure with --with-cudadecoder[=true]")
endif

endif

include ../makefiles/default_rules.mk
