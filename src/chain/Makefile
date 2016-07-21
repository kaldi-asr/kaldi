
all:

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = chain-supervision-test language-model-test

OBJFILES = chain-supervision.o chain-numerator.o chain-den-graph.o \
          language-model.o chain-denominator.o chain-training.o
ifeq ($(CUDA), true)
  OBJFILES += chain-kernels.o
endif

LIBNAME = kaldi-chain

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

# Make sure we have CUDA_ARCH from kaldi.mk,
ifeq ($(CUDA), true)
  ifndef CUDA_ARCH
    $(error CUDA_ARCH is undefined, run 'src/configure')
  endif
endif

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../

include ../makefiles/default_rules.mk

