

all:

include ../kaldi.mk
ifeq ($(CUDA), true)

TESTFILES = 


OBJFILES +=  feature-window-cuda.o feature-mfcc-cuda.o


LIBNAME = kaldi-cudafeat

ADDLIBS = ../feat/kaldi-feat.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../cudamatrix/kaldi-cudamatrix.a \
          ../gmm/kaldi-gmm.a 

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c -g $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../ -I$(OPENFSTINC)
endif

include ../makefiles/default_rules.mk
