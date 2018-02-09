
all:

EXTRA_CXXFLAGS = -I$(PCREROOT)/include -I$(EXPATROOT)/include -I$(PUJIXMLROOT)/src
EXTRA_LDLIBS = $(PCREROOT)/lib/libpcre.a $(EXPATROOT)/lib/libexpat.a $(PUJIXMLROOT)/scripts/libpugixml.a
OPENFST_CXXFLAGS =
OPENFST_LDLIBS =

include ../kaldi.mk

BINFILES = idlaktxp idlakcex

OBJFILES =

TESTFILES =

ADDLIBS = ../idlaktxp/idlak-txp.a ../tree/kaldi-tree.a  ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

