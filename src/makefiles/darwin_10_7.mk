# makefiles/darwin_10_6.mk contains Darwin-specific rules for OS X 10.7.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

DOUBLE_PRECISION = 0
CXXFLAGS += -msse -msse2 -Wall -I.. \
	  -pthread \
      -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -rdynamic \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID


ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -g -rdynamic
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar
