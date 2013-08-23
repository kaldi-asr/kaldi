

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = cu-matrix-test cu-test cu-sp-matrix-test cu-packed-matrix-test cu-tp-matrix-test cu-vector-test


OBJFILES = cu-device.o cu-math.o cu-matrix.o cu-packed-matrix.o cu-sp-matrix.o cu-vector.o cu-common.o cu-tp-matrix.o
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o cu-randkernels.o cu-choleskykernels.o
endif

LIBNAME = kaldi-cudamatrix

all:  $(LIBFILE)


ifeq ($(CUDA), true)
  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_10,code=sm_10 \
	    -gencode arch=compute_13,code=sm_13 \
	    -gencode arch=compute_20,code=sm_20
  #Get the CUDA Toolkit version (remove decimal point char)
  CUDA_VERSION=$(shell $(CUDATKDIR)/bin/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
  #For toolkit 4.2 or newer, add the compute capability 3.0 
  CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
  ifeq ($(CUDA_VER_GT_4_2), true)
    CUDA_ARCH += -gencode arch=compute_30,code=sm_30
  endif
endif


#implicit rule for kernel compilation
#symlink in necessary for cross-compilation at BUT
%.o : %.cu
	-ln -s `which $(CC)` gcc
	export PATH=$(CUDATKDIR)/bin:$(PWD):$(PWD)/cudamatrix:$(PATH); $(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../
	-unlink gcc


ADDLIBS = ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

