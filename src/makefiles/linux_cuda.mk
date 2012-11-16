
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath=$(CUDATKDIR)/lib
LDFLAGS += -lcublas -lcudart -lcuda

