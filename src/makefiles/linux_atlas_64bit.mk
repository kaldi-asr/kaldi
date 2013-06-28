# This version is specialized for 64-bit cross-compilation on BUT machines.
# The configure script will not pick it up automatically.
#
# To use it, from src/ run in bash:
#
# cat makefiles/common.mk makefiles/linux_atlas_64bit.mk > kaldi.mk
# echo "CUDA = true" >> kaldi.mk
# echo "CUDATKDIR = /usr/local/share/cuda" >> kaldi.mk
# cat makefiles/linux_x86_64_cuda.mk >> kaldi.mk
#
# Note that for 64bit compilation of kaldi, 
# you need to compile 64bit OpenFST first.
#

### You need to set KALDI_ROOT manually
KALDI_ROOT=/mnt/matylda5/iveselyk/DEVEL/kaldi/trunk
###

FSTROOT = $(KALDI_ROOT)/tools/openfst
ATLASINC = $(KALDI_ROOT)/tools/ATLAS/include
ATLASLIBS = /usr/local/lib64/liblapack.a /usr/local/lib64/libcblas.a /usr/local/lib64/libatlas.a /usr/local/lib64/libf77blas.a

# You have to make sure ATLASLIBS is set...

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef ATLASINC
$(error ATLASINC not defined.)
endif

ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif


CXXFLAGS = -msse -msse2 -Wall -I.. \
	  -fPIC \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I$(ATLASINC) \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -rdynamic $(OPENFSTLDFLAGS)
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(ATLASLIBS) -lm -lpthread -ldl 
CC = x86_64-linux-g++
CXX = x86_64-linux-g++
AR = x86_64-linux-ar
AS = x86_64-linux-as
RANLIB = x86_64-linux-ranlib
