all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
           lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
           decoder-wrappers.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

# for GPU-bsed WFST decoder
ifeq ($(BUILD_CUDA_DECODER), true)
OBJFILES += cuda-decoder-utils.o cuda-lattice-decoder.o \
            lattice-faster-decoder-cuda.o cuda-decoder.o 
LDLIBS += $(CUDA_LDLIBS)
LDFLAGS += $(CUDA_LDFLAGS)
ADDLIBS += ../cudamatrix/kaldi-cudamatrix.a

# Unfortunately, this code depends on 64-bit atomics, which are not
# provided in compute capability 3.0 (although that compute capability
# is otherwise suppored within Kaldi). See here:
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicmin
# If you run the CUDA decoder on a device of compute-capability 3.0, you will
# get an error when you try to launch the relevant kernels.
comma := ,
CUDA_ARCH_NEW_ENOUGH = $(subst -gencode arch=compute_30$(comma)code=sm_30,,$(CUDA_ARCH))

# Why in the world is -w here??? Seems terrible!
CUDA_DEC_OPTS = -lineinfo  #-w

# TODO: Tricky bit: Some parts of CXXFLAGS need to go into -Xcompiler... Hmmm...
# CUDA_DEC_OPTS = -lineinfo -w -Xcompiler "-msse -msse2 -pthread"

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) \
	$(CUDA_ARCH_NEW_ENOUGH) -I../ -isystem $(OPENFSTINC) $(CUDA_DEC_OPTS)
endif

include ../makefiles/default_rules.mk
