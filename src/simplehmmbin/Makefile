
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = simple-hmm-init \
					 compile-train-simple-hmm-graphs simple-hmm-align-compiled \
					 simple-hmm-acc-stats-ali simple-hmm-est make-simple-hmm-graph


OBJFILES =

ADDLIBS = ../decoder/kaldi-decoder.a \
					../simplehmm/kaldi-simplehmm.a ../lat/kaldi-lat.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a


TESTFILES =

include ../makefiles/default_rules.mk

