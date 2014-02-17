all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare
EXTRA_CXXFLAGS += -ggdb -O0  # for gdb debugging, example -DDEBUG_POST for MovePostToArcs debug

# Kaldi libraries required
ADDLIBS = ../decoder/kaldi-decoder.a ../thread/kaldi-thread.a \
		../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
		../fstext/kaldi-fstext.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
		../feat/kaldi-feat.a ../util/kaldi-util.a ../base/kaldi-base.a

BINFILES =

TESTFILES = dec-wrap-feat-input-test \
			dec-wrap-audio-source-test read-wav-test dec-wrap-utils-test

OBJFILES = dec-wrap-latgen-decoder.o dec-wrap-latgen-wrapper.o \
		   dec-wrap-audio-source.o  dec-wrap-utils.o \
		   dec-wrap-decodable.o dec-wrap-feat-input.o

LIBNAME = dec-wrap


include ../makefiles/default_rules.mk

cleantest:
	rm -f *.txt *.svg *.fst

cleanall: clean cleantest
