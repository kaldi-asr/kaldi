
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
UNAME := $(shell uname)
#aware of fact in cuda60 there is no lib64, just lib.
ifeq ($(UNAME), Darwin)
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
else
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
endif
CUDA_LDLIBS += -lcublas -lcudart #LDLIBS : The libs are loaded later than static libs in implicit rule

