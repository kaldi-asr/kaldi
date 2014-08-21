# makefiles/darwin_10_5.mk contains Darwin-specific rules for OS X 10.5.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -I.. \
	  -pthread \
      -DKALDI_DOUBLEPRECISION=0  \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -gdwarf-2 # -O0 -DKALDI_PARANOID

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -gdwarf-2
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
CXX = g++-4
CC = g++-4
RANLIB = ranlib
AR = ar
