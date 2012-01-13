
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
LDFLAGS += -lcublas -lcudart -lcuda

