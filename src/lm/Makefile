EXTRA_CXXFLAGS = -Wno-sign-compare

all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
MATHLIB = NONE

# Uncomment following line to use IRSTLM toolkit installed in ../lmtoolkit
#include ./irstlm.mk

include ../kaldi.mk

TESTFILES = lm-lib-test

OBJFILES = const-arpa-lm.o kaldi-lmtable.o kaldi-lm.o

TESTOUTPUTS = composed.fst output.fst output1.fst output2.fst

LIBNAME = kaldi-lm

ADDLIBS = ../base/kaldi-base.a ../fstext/kaldi-fstext.a ../util/kaldi-util.a

include ../makefiles/default_rules.mk
