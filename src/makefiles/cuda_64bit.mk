ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
$(error CUDATKDIR not defined.)
endif
ifndef CUDNNDIR
$(error CUDNNDIR not defined.)
endif

# Order matters here. We must tell the compiler to search
# $(CUDNNDIR)/lib64 before $(CUDATKDIR)/lib64 because the CUDNN .deb
# files install cudnn to /usr/local/cuda/lib64, which would overshadow
# the user-specified $(CUDNNDIR)
CUDA_INCLUDE += -I$(CUDNNDIR)/include
CXXFLAGS += -I$(CUDNNDIR)/include
CUDA_LDFLAGS += -L$(CUDNNDIR)/lib64 -Wl,-rpath,$(CUDNNDIR)/lib64
CUDA_LDLIBS += -lcudnn

CUDA_INCLUDE= -I$(CUDATKDIR)/include -I$(CUBROOT)
CUDA_FLAGS = -Xcompiler "-fPIC -pthread -isystem $(OPENFSTINC)" --verbose --machine 64 -DHAVE_CUDA=1 \
             -ccbin $(CXX) -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
             -std=c++11 -DCUDA_API_PER_THREAD_DEFAULT_STREAM  -I$(CUDATKDIR)/include

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include

CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcublas -lcusparse -lcudart -lcurand -lnvToolsExt #LDLIBS : The libs are loaded later than static libs in implicit rule
