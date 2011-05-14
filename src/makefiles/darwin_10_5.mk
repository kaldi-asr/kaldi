# makefiles/darwin_10_5.mk contains Darwin-specific rules for OS X 10.5.*

CXXFLAGS = -msse -msse2 -Wall -I.. -DKALDI_DOUBLEPRECISION=0  \
    -DHAVE_CLAPACK \
    -Wno-sign-compare -Winit-self \
    -I../../tools/openfst/include \
    -DHAVE_EXECINFO_H -DHAVE_CXXABI_H \
    -g -O0 -DKALDI_PARANOID

LDFLAGS = -g
LDLIBS = ../../tools/openfst/lib/libfst.a -ldl -lm -framework Accelerate
CXX = g++-4
CC = g++-4
RANLIB = ranlib
AR = ar
