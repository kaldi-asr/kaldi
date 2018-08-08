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
ifeq ($(CUDA), true)
OBJFILES += cuda-decoder-utils.o cuda-lattice-decoder.o \
            lattice-faster-decoder-cuda.o cuda-decoder.o 
LDLIBS += $(CUDA_LDLIBS)
LDFLAGS += $(CUDA_LDFLAGS)
ADDLIBS += ../cudamatrix/kaldi-cudamatrix.a

CUDAOPTS = -O3 $(CUDA_ARCH) 
#CUDAOPTS += -G  
CUDA_DEC_OPTS = -DHAVE_CXXABI_H -lineinfo -w -DHAVE_EXECINFO_H=1 \
               -DHAVE_ATLAS -Xcompiler "-msse -msse2 -pthread" \
               $(CUDAOPTS) -I.. -I$(OPENFSTINC) 

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) \
	 -I../ $(CUDA_DEC_OPTS)
endif

include ../makefiles/default_rules.mk
