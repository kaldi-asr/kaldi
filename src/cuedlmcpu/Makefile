EXTRA_CXXFLAGS = -Wno-sign-compare -fopenmp


all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
MATHLIB = NONE

# Uncomment following line to use IRSTLM toolkit installed in ../lmtoolkit

include ../kaldi.mk

TESTFILES =

OBJFILES = cued-rnnlm-lib.o  fileops.o  helper.o  Mathops.o

TESTOUTPUTS =

LIBNAME = cued-lm

ADDLIBS =

include ../makefiles/default_rules.mk
