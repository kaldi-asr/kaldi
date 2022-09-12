ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef ROCMDIR
$(error ROCMDIR not defined.)
endif

# Uncomment if willing to use ROCTX capabilities.
# ROCM_USEROCTX = -DUSE_NVTX

# Specific HIP/ROCm components should be included prior to the generic include to avoid
# deprecation warnings.
CXXFLAGS += $(ROCM_USEROCTX) -DHAVE_CUDA=1 \
            -D__IS_HIP_COMPILE__=1 \
            -DROCM_MAJOR_VERSION=$(ROCM_MAJOR_VERSION) -DROCM_MINOR_VERSION=$(ROCM_MINOR_VERSION) \
            -DCUDA_VERSION=11000 \
	          -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -I$(ROCMDIR)/include -I../hip -fPIC -pthread -isystem $(OPENFSTINC)

ROCM_INCLUDE = -I$(ROCMDIR)/hiprand/include -I$(ROCMDIR)/rocrand/include -I$(ROCMDIR)/include -I../hip -isystem $(OPENFSTINC)
ROCM_FLAGS = $(ROCM_USEROCTX) -fPIC -DHAVE_CUDA=1 \
             -D__IS_HIP_COMPILE__=1 \
             -DROCM_MAJOR_VERSION=$(ROCM_MAJOR_VERSION) -DROCM_MINOR_VERSION=$(ROCM_MINOR_VERSION) \
             -D__CUDACC_VER_MAJOR__=11 -D__CUDA_ARCH__=800 -DCUDA_VERSION=11000 \
	           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) -std=c++14

#TODO: Consider use ROCM_LDFLAGS/ROCM_LDLIBS or generic GPU_LDFLAGS/GPU_LDLIBS in the makefiles.
CUDA_LDFLAGS += -L$(ROCMDIR)/lib -Wl,-rpath,$(ROCMDIR)/lib
CUDA_LDLIBS += -lhipblas -lhipsparse -lhipsolver -lhiprand -lamdhip64
