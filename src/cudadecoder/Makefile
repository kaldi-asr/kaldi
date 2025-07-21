all: ;

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

ifeq ($(CUDA), true)
ifeq ($(WITH_CUDADECODER), true)

# Make sure we have CUDA_ARCH from kaldi.mk,
ifndef CUDA_ARCH
  $(error CUDA_ARCH is undefined, run 'src/configure')
endif

TESTFILES =

OBJFILES = cuda-decoder.o cuda-decoder-kernels.o cuda-fst.o \
	   batched-threaded-nnet3-cuda-online-pipeline.o \
	   batched-threaded-nnet3-cuda-pipeline.o \
	   batched-threaded-nnet3-cuda-pipeline2.o \
	   batched-static-nnet3.o batched-static-nnet3-kernels.o \
	   cuda-online-pipeline-dynamic-batcher.o decodable-cumatrix.o \
	   cuda-pipeline-common.o lattice-postprocessor.o

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
