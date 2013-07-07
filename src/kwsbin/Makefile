
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-to-kws-index kws-index-union transcripts-to-fsts \
		   kws-search generate-proxy-keywords

OBJFILES =



TESTFILES =


ADDLIBS = ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a \
        ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
        ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
