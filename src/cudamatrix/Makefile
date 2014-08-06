

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = cu-vector-test cu-matrix-test cu-math-test cu-test cu-sp-matrix-test cu-packed-matrix-test cu-tp-matrix-test \
            cu-block-matrix-test cu-matrix-speed-test cu-vector-speed-test cu-sp-matrix-speed-test cu-array-test


OBJFILES = cu-device.o cu-math.o cu-matrix.o cu-packed-matrix.o cu-sp-matrix.o \
           cu-vector.o cu-common.o cu-tp-matrix.o cu-rand.o cu-block-matrix.o
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o cu-randkernels.o
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
endif


#implicit rule for kernel compilation
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../


ADDLIBS = ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

