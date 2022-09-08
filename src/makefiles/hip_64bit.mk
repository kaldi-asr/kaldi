ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef ROCMDIR
$(error ROCMDIR not defined.)
endif

CXXFLAGS += -DHAVE_CUDA=1 -D__IS_HIP_COMPILE__=1 -D__HIP_PLATFORM_AMD__=1 -DCUDA_VERSION=11000 \
	    -I$(ROCMDIR)/include -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -I../hip -fPIC -pthread -isystem $(OPENFSTINC)

ROCM_INCLUDE= -I$(ROCMDIR)/include -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -I.. -I../hip -isystem $(OPENFSTINC)
ROCM_FLAGS = -fPIC -DHAVE_CUDA=1 \
             -D__IS_HIP_COMPILE__=1 -D__CUDACC_VER_MAJOR__=11 -D__CUDA_ARCH__=800 -DCUDA_VERSION=11000 \
	     -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) -std=c++14 -fgpu-default-stream=per-thread

#CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64/stubs -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
#CUDA_LDFLAGS += -L$(CUDATKDIR)/lib/stubs -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
ROCM_LDFLAGS += 

#CUDA_LDLIBS += -lcuda -lcublas -lcusparse -lcusolver -lcudart -lcurand -lcufft -lnvToolsExt
ROCM_LDLIBS += 
