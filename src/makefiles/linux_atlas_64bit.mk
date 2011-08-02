# This version is specialized for 64-bit cross-compilation on BUT machines.
# The configure script will not pick it up automatically.
# To use it: from .., do:
# cat makefiles/kaldi.mk.common makefiles/kaldi.mk.linux.64bit > kaldi.mk


CXXFLAGS = -msse -msse2 -Wall -I.. -DKALDI_DOUBLEPRECISION=0 \
      -Wno-sign-compare -Winit-self \
      -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I ../../tools/ATLAS/include -I../../tools/openfst/include \
      -g -O0 -DKALDI_PARANOID 

LDFLAGS = -rdynamic
LDLIBS = ../../tools/openfst/lib/libfst.a -ldl /usr/local/lib64/liblapack.a /usr/local/lib64/libcblas.a /usr/local/lib64/libatlas.a -lg2c -lm
CC = x86_64-linux-g++
CXX = x86_64-linux-g++
AR = x86_64-linux-ar
AS = x86_64-linux-as
RANLIB = x86_64-linux-ranlib
