ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
$(error CUDATKDIR not defined.)
endif

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include -fPIC -pthread -isystem $(OPENFSTINC)

CUDA_INCLUDE= -I$(CUDATKDIR)/include -I$(CUBROOT) -I.. -isystem $(OPENFSTINC)
CUDA_FLAGS = --compiler-options -fPIC --machine 64 -DHAVE_CUDA \
             -ccbin $(CXX) -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
             -std=c++14 -DCUDA_API_PER_THREAD_DEFAULT_STREAM -lineinfo \
             --verbose -Wno-deprecated-gpu-targets

CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64/stubs -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib/stubs -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib

CUDA_LDLIBS += -lcuda -lcublas -lcusparse -lcusolver -lcudart -lcurand -lcufft -lnvToolsExt
