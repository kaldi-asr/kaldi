
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare


OBJFILES = kws-functions.o kws-functions2.o kws-scoring.o
LIBNAME = kaldi-kws

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 


include ../makefiles/default_rules.mk
