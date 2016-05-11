
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare


OBJFILES = kws-functions.o kws-functions2.o kws-scoring.o
LIBNAME = kaldi-kws

ADDLIBS = ../hmm/kaldi-hmm.a ../lat/kaldi-lat.a ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../base/kaldi-base.a


include ../makefiles/default_rules.mk
