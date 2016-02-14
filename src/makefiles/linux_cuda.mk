
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA

CXXFLAGS := -DHAVE_CUDA -I$(CUDATKDIR)/include $(CXXFLAGS)
LDFLAGS := -L$(CUDATKDIR)/lib -Wl,-rpath=$(CUDATKDIR)/lib $(LDFLAGS)
# LDLIBS : The libs are loaded later than static libs in implicit rule
LDLIBS := -lcublas -lcudart $(LDLIBS)

