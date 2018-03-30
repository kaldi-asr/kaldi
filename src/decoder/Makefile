all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += -lnvToolsExt -lcudart
CUDAOPTS = -O3 $(CUDA_ARCH) 
#CUDAOPTS += -G  
CUDA_DC_OPTS = -dc -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DHAVE_CXXABI_H \
               -lineinfo -w -std=c++11 -DHAVE_EXECINFO_H=1 \
               -DHAVE_ATLAS -Xcompiler "-msse -msse2 -pthread  -fopenmp" \
               $(CUDAOPTS) -I.. -I$(OPENFSTINC) 

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
           lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
           decoder-wrappers.o \
           cuda-decoder.o cuda-decoder-utils_prelink.o cuda-lattice-decoder_prelink.o \
           lattice-faster-decoder-cuda.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

cuda-decoder_test.o: cuda-decoder_test.cu
	$(CUDATKDIR)/bin/nvcc  -G -g -O0 -v -c $(CUDA_ARCH) cuda-decoder_test.cu 

cuda-decoder_test:cuda-decoder_test.o
	g++ -g -O0 -o cuda-decoder_test $(CUDA_LDFLAGS)  $(CUDA_LDLIBS)  -lcuda cuda-decoder_test.o

cuda-lattice-decoder_prelink.o: cuda-lattice-decoder.cu cuda-lattice-decoder.h cuda-decoder-utils.h
	$(CUDATKDIR)/bin/nvcc $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_DC_OPTS) 
cuda-decoder-utils_prelink.o: cuda-decoder-utils.cu cuda-decoder-utils.h
	$(CUDATKDIR)/bin/nvcc $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_DC_OPTS) 

cuda-decoder.o:  cuda-decoder-utils_prelink.o  cuda-lattice-decoder_prelink.o
	$(CUDATKDIR)/bin/nvcc $^ -o $@ -rdc=true -dlink -Xcompiler "-pthread -fPIC" -v $(CUDAOPTS) 

include ../makefiles/default_rules.mk
