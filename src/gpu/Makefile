
all:

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

OBJFILES = 
ifeq ($(CUDA), true)
  OBJFILES += gpu-diag-gmm.o gpu-transition-model.o gpu-am-diag-gmm.o gpu-online-decodable.o
endif

LIBNAME = kaldi-gpu

ADDLIBS = ../util/kaldi-util.a \
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
