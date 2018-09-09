ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
$(error CUDATKDIR not defined.)
endif

# Order matters here. We must tell the compiler to search
# $(CUDNNDIR)/lib64 before $(CUDATKDIR)/lib64 because the CUDNN .deb
# files install cudnn to /usr/local/cuda/lib64, which would overshadow
# the user-specified $(CUDNNDIR)
ifdef CUDNNDIR
CUDA_INCLUDE += -I$(CUDNNDIR)/include
CUDA_FLAGS += -DHAVE_CUDNN=1
CXXFLAGS += -I$(CUDNNDIR)/include -DHAVE_CUDNN=1
CUDA_LDFLAGS += -L$(CUDNNDIR)/lib64 -Wl,-rpath,$(CUDNNDIR)/lib64
CUDA_LDLIBS += -lcudnn
endif
CUDA_INCLUDE += -I$(CUDATKDIR)/include
CUDA_FLAGS += -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA=1 \
						 -ccbin $(CXX) \
						 -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
             -DCUDA_API_PER_THREAD_DEFAULT_STREAM
CXXFLAGS += -DHAVE_CUDA=1 -I$(CUDATKDIR)/include
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcublas -lcusparse -lcudart -lcurand #LDLIBS : The libs are loaded later than static libs in implicit rule
