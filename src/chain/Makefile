
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

ifeq ($(CUDA), true)
  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_20,code=sm_20
  #Get the CUDA Toolkit version (remove decimal point char)
  CUDA_VERSION=$(shell $(CUDATKDIR)/bin/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
  #For toolkit 4.2 or newer, add the compute capability 3.0
  CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
  ifeq ($(CUDA_VER_GT_4_2), true)
    CUDA_ARCH += -gencode arch=compute_30,code=sm_30
  endif
  #For toolkit 5.0 or newer, add the compute capability 3.5
  CUDA_VER_GT_5_0 := $(shell [ $(CUDA_VERSION) -ge 50 ] && echo true)
  ifeq ($(CUDA_VER_GT_5_0), true)
    CUDA_ARCH += -gencode arch=compute_35,code=sm_35
  endif
  #For toolkit 6.0 or newer, add the compute capability 5.0
  CUDA_VER_GT_6_0 := $(shell [ $(CUDA_VERSION) -ge 60 ] && echo true)
  ifeq ($(CUDA_VER_GT_6_0), true)
    CUDA_ARCH += -gencode arch=compute_50,code=sm_50
  endif
  #For toolkit older than 6.5, add the compute capability 1.0
  CUDA_VER_GT_6_5 := $(shell [ $(CUDA_VERSION) -ge 65 ] && echo true)
  ifneq ($(CUDA_VER_GT_6_5), true)
    CUDA_ARCH += -gencode arch=compute_13,code=sm_13 \
                 -gencode arch=compute_10,code=sm_10
  endif
endif


#implicit rule for kernel compilation
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../


ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
           ../fstext/kaldi-fstext.a \
           ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk

