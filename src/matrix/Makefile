

all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =

include ../kaldi.mk


# you can uncomment matrix-lib-speed-test if you want to do the speed tests.

TESTFILES = matrix-lib-test sparse-matrix-test numpy-array-test #matrix-lib-speed-test

OBJFILES = kaldi-matrix.o kaldi-vector.o packed-matrix.o sp-matrix.o tp-matrix.o \
           matrix-functions.o qr.o srfft.o compressed-matrix.o \
           sparse-matrix.o optimization.o numpy-array.o

LIBNAME = kaldi-matrix

ADDLIBS = ../base/kaldi-base.a

include ../makefiles/default_rules.mk

