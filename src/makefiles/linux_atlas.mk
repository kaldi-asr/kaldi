# You have to make sure ATLASLIBS is set...

CXXFLAGS = -msse -Wall -I.. \
      -DKALDI_DOUBLEPRECISION=0 -msse2 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I ../../tools/ATLAS/include \
      -I ../../tools/openfst/include \
      -g -O0 -DKALDI_PARANOID 

LDFLAGS = -rdynamic
LDLIBS = ../../tools/openfst/lib/libfst.a -ldl $(ATLASLIBS) -lm
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
