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


DOUBLE_PRECISION = 0
CXXFLAGS = -m64 -maltivec -mcpu=power8 -Wall -I.. \
	   -mtune=power8 -mpower8-vector -mvsx -pthread \
      -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
      -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
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
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
