# makefiles/kaldi.mk.darwin contains Darwin-specific rules

CXXFLAGS = -msse -msse2 -Wall -I.. -DKALDI_DOUBLEPRECISION=0  \
    -DHAVE_POSIX_MEMALIGN -DHAVE_CLAPACK \
    -Wno-sign-compare -Winit-self \
    -I../../tools/openfst/include \
    -DHAVE_EXECINFO_H -DHAVE_CXXABI_H \
    -rdynamic \
    -g -O0 -DKALDI_PARANOID

LDFLAGS = -g -rdynamic
LDLIBS = ../../tools/openfst/lib/libfst.a -ldl -lm -framework Accelerate
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar
