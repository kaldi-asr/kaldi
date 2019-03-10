all:


include ../kaldi.mk

TESTFILES = topology-test hmm-utils-test transitions-test posterior-test

OBJFILES = topology.o transitions.o hmm-utils.o tree-accu.o \
        posterior.o hmm-test-utils.o

LIBNAME = kaldi-hmm
ADDLIBS = ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
