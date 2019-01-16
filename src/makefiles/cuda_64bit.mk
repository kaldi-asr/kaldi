ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
$(error CUDATKDIR not defined.)
endif

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include -fPIC -pthread -isystem $(OPENFSTINC) -rdynamic

CUDA_INCLUDE= -I$(CUDATKDIR)/include -I$(CUBROOT)
CUDA_FLAGS = --machine 64 -DHAVE_CUDA \
             -ccbin $(CXX) -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
             -std=c++11 -DCUDA_API_PER_THREAD_DEFAULT_STREAM  -lineinfo \
             --verbose -Xcompiler "$(CXXFLAGS)"

CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcublas -lcusparse -lcudart -lcurand -lnvToolsExt #LDLIBS : The libs are loaded later than static libs in implicit rule
