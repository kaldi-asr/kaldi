# You have to make sure CLAPACKLIBS is set...

KALDI_CXXFLAGS = -msse -Wall -I.. -pthread \
      -DKALDI_DOUBLEPRECISION=0 -msse2 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Wno-unused-local-typedefs \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_CLAPACK -I ../../tools/CLAPACK \
      -I ../../tools/openfst/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
KALDI_CXXFLAGS += -fPIC
endif

CXXFLAGS := $(KALDI_CXXFLAGS) $(CXXFLAGS)
LDFLAGS := -rdynamic $(OPENFSTLDFLAGS) $(LDFLAGS)
LDLIBS := $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(ATLASLIBS) -lm -lpthread -ldl $(LDLIBS)
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
