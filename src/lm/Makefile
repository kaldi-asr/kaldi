EXTRA_CXXFLAGS = -Wno-sign-compare

all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
MATHLIB = NONE

# Uncomment following line to use IRSTLM toolkit installed in ../lmtoolkit
#include ./irstlm.mk

include ../kaldi.mk

TESTFILES = lm-lib-test

OBJFILES = kaldi-lmtable.o kaldi-lm.o

TESTOUTPUTS = composed.fst output.fst output1.fst output2.fst

LIBNAME = kaldi-lm

ADDLIBS = ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
