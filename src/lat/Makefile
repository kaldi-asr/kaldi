
all:

include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare

TESTFILES = kaldi-lattice-test

OBJFILES = kaldi-lattice.o lattice-functions.o word-align-lattice.o \
	   phone-align-lattice.o word-align-lattice-lexicon.o sausages.o \
           kws-functions.o push-lattice.o

LIBNAME = kaldi-lat

ADDLIBS = ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
