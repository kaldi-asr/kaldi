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
            -D__HIP_PLATFORM_AMD__=1 \
            -D__IS_HIP_COMPILE__=1 \
            -DROCM_MAJOR_VERSION=$(ROCM_MAJOR_VERSION) -DROCM_MINOR_VERSION=$(ROCM_MINOR_VERSION) \
            -DCUDA_VERSION=11000 \
	          -I$(ROCMDIR)/hipsparse/include \
	          -I$(ROCMDIR)/hipfft/include \
	          -I$(ROCMDIR)/hipblas/include \
	          -I$(ROCMDIR)/hiprand/include \
	          -I$(ROCMDIR)/rocrand/include \
	          -I$(ROCMDIR)/include \
	          -I.. -I../hip -fPIC -pthread -isystem $(OPENFSTINC)

ROCM_INCLUDE = -I$(ROCMDIR)/hipsparse/include \
               -I$(ROCMDIR)/hipfft/include \
               -I$(ROCMDIR)/hipblas/include \
               -I$(ROCMDIR)/hiprand/include \
               -I$(ROCMDIR)/rocrand/include \
               -I$(ROCMDIR)/include \
               -I.. -I../hip -isystem $(OPENFSTINC)
               
# TODO: Consider passing __CUDA_ARCH__=800 here as it is mostly supported by ROCm.
#       However this macro has some side effect with HIPCC that makes it assume
#       CUDA is active and everything is device compiles.
ROCM_FLAGS = $(ROCM_USEROCTX) -fPIC -DHAVE_CUDA=1 \
             -D__IS_HIP_COMPILE__=1 \
             -D__HIP_PLATFORM_AMD__=1 \
             -DROCM_MAJOR_VERSION=$(ROCM_MAJOR_VERSION) -DROCM_MINOR_VERSION=$(ROCM_MINOR_VERSION) \
             -D__CUDACC_VER_MAJOR__=11 -DCUDA_VERSION=11000 \
	         -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) -std=c++14 -munsafe-fp-atomics  \
             -fgpu-default-stream=per-thread \
             $(EXTRA_ROCM_FLAGS)
             

# TODO: Consider use ROCM_LDFLAGS/ROCM_LDLIBS or generic GPU_LDFLAGS/GPU_LDLIBS in the makefiles.
# We allow the libraries we link against to have undefined symbols so as this can be build in
# systems with no development version of these libraries (e.g. ncurses).
CUDA_LDFLAGS += -L$(ROCMDIR)/lib -Wl,-rpath,$(ROCMDIR)/lib
CUDA_LDLIBS += -lhipblas -lhipsparse -lhipsolver -lhiprand -lhipfft -lroctx64 -lamdhip64 -Wl,--allow-shlib-undefined 
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
