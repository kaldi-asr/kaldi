all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
           lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
           decoder-wrappers.o 

ifeq ($(CUDA), true)
OBJFILES += cuda-decoder-utils.o cuda-lattice-decoder.o \
            lattice-faster-decoder-cuda.o
LDLIBS += -lnvToolsExt -lcudart
LDFLAGS += $(CUDA_LDFLAGS)
LDFLAGS += ../cudamatrix/libkaldi-cudamatrix.so
LDLIBS += $(CUDA_LDLIBS)
endif

CUDAOPTS = -O3 $(CUDA_ARCH) 
#CUDAOPTS += -G  
CUDA_DEC_OPTS = -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DHAVE_CXXABI_H \
               -lineinfo -w -std=c++11 -DHAVE_EXECINFO_H=1 \
               -DHAVE_ATLAS -Xcompiler "-msse -msse2 -pthread  -fopenmp" \
               $(CUDAOPTS) -I.. -I$(OPENFSTINC) 

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) \
	 -I../ $(CUDA_DEC_OPTS)

include ../makefiles/default_rules.mk
