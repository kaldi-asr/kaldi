
all:


include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

OBJFILES =
OBJNONLINKEDFILES =

ifeq ($(CUDA), true)
  OBJNONLINKEDFILES += gpu-diag-gmm.o gpu-transition-model.o gpu-am-diag-gmm.o gpu-online-decodable.o
  OBJFILES += $(OBJNONLINKEDFILES)
  OBJFILES += kaldi-gpu.dlink.o
endif

LIBNAME = kaldi-gpu

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

# Make sure we have CUDA_ARCH from kaldi.mk,
ifeq ($(CUDA), true)
  ifndef CUDA_ARCH
    $(error CUDA_ARCH is undefined, run 'src/configure')
  endif
endif

%.dlink.o :
	$(CUDATKDIR)/bin/nvcc $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -dlink -o $@ $(OBJNONLINKEDFILES)

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -dc $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../ -std=c++11

include ../makefiles/default_rules.mk
