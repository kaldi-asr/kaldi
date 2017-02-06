all:


include ../kaldi.mk

TESTFILES = simple-hmm-test

OBJFILES = simple-hmm.o simple-hmm-utils.o

LIBNAME = kaldi-simplehmm
ADDLIBS = ../hmm/kaldi-hmm.a ../decoder/kaldi-decoder.a \
					../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk

