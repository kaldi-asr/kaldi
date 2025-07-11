
all:

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = cu-vector-test cu-matrix-test cu-math-test cu-test cu-sp-matrix-test cu-packed-matrix-test cu-tp-matrix-test \
            cu-block-matrix-test cu-matrix-speed-test cu-vector-speed-test cu-sp-matrix-speed-test cu-array-test \
	    cu-sparse-matrix-test cu-device-test cu-rand-speed-test cu-compressed-matrix-test

OBJFILES = cu-device.o cu-math.o cu-rand.o cu-matrix.o cu-packed-matrix.o cu-sp-matrix.o \
           cu-vector.o cu-common.o cu-tp-matrix.o cu-block-matrix.o \
           cu-sparse-matrix.o cu-allocator.o cu-array.o cu-compressed-matrix.o
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o
endif

LIBNAME = kaldi-cudamatrix

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

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
