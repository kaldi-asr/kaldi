all:
		
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

ifeq ($(CUDA), true)
ifneq ($(WITH_CUDADECODER), 0)

# Make sure we have CUDA_ARCH from kaldi.mk,
ifndef CUDA_ARCH
  $(error CUDA_ARCH is undefined, run 'src/configure')
endif

TESTFILES =

OBJFILES = batched-threaded-nnet3-cuda-pipeline.o decodable-cumatrix.o \
           cuda-decoder.o cuda-decoder-kernels.o cuda-fst.o

LIBNAME = kaldi-cudadecoder

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../base/kaldi-base.a ../matrix/kaldi-matrix.a \
          ../lat/kaldi-lat.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../gmm/kaldi-gmm.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../transform/kaldi-transform.a \
          ../tree/kaldi-tree.a ../online2/kaldi-online2.a ../nnet3/kaldi-nnet3.a \
					../cudafeat/kaldi-cudafeat.a

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


# Implicit rule for kernel compilation
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../ -I$(OPENFSTINC)

else
all:
		$(warning "Not building cudadecoder extension -- to build with it, configure with --with-cudadecoder[=true]")
endif

endif

include ../makefiles/default_rules.mk
