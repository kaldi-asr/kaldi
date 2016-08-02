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


DOUBLE_PRECISION = 0
CXXFLAGS = -ftree-vectorize -mfloat-abi=hard -mfpu=neon -Wall -I.. \
           -pthread \
      -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
      -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_OPENBLAS -I $(OPENBLASROOT)/include \
      -I $(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -rdynamic $(OPENFSTLDFLAGS)
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(OPENBLASLIBS) -lm -lpthread -ldl 
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
