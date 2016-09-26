ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif


CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA \
             -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION)
CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include
LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath=$(CUDATKDIR)/lib
LDLIBS += -lcublas -lcudart -lcurand #LDLIBS : The libs are loaded later than static libs in implicit rule
