# You have to make sure FSTROOT,OPENBLASROOT,OPENBLASLIBS are set...

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef OPENBLASLIBS
$(error OPENBLASLIBS not defined.)
endif

ifndef OPENBLASROOT
$(error OPENBLASROOT not defined.)
endif


KALDI_CXXFLAGS = -msse -msse2 -Wall -I.. \
           -pthread \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_OPENBLAS -I $(OPENBLASROOT)/include \
      -I $(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
KALDI_CXXFLAGS += -fPIC
endif

CXXFLAGS := $(KALDI_CXXFLAGS) $(CXXFLAGS)
LDFLAGS := -rdynamic $(OPENFSTLDFLAGS) $(LDFLAGS)
LDLIBS := $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(OPENBLASLIBS) -lm -lpthread -ldl $(LDLIBS)
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
