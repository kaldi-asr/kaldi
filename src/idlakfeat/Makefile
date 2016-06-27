

all:

include ../kaldi.mk

TESTFILES = feature-aperiodic-test

OBJFILES = feature-mcep.o banks-computations.o feature-aperiodic.o minmax.o hmm-utils-idlak.o

LIBNAME = idlak-feat

ADDLIBS = ../transform/kaldi-transform.a ../hmm/kaldi-hmm.a \
	../util/kaldi-util.a ../thread/kaldi-thread.a ../feat/kaldi-feat.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

