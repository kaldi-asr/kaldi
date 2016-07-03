all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   lattice-tracking-decoder.o decoder-wrappers.o

LIBNAME = kaldi-decoder

ADDLIBS = ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../lat/kaldi-lat.a \
     ../sgmm/kaldi-sgmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../util/kaldi-util.a \
     ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

