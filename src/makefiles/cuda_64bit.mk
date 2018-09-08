ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
$(error CUDATKDIR not defined.)
endif

# TODO: Clean this up to make cudnn an optional dependency
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA=1
						 -DHAVE_CUDNN=1 -ccbin $(CXX) \
						 -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
             -DCUDA_API_PER_THREAD_DEFAULT_STREAM
CXXFLAGS += -DHAVE_CUDA=1 -DHAVE_CUDNN=1 -I$(CUDATKDIR)/include
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcudnn -lcublas -lcusparse -lcudart -lcurand #LDLIBS : The libs are loaded later than static libs in implicit rule
