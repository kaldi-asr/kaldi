
all:

EXTRA_CXXFLAGS = -Wno-sign-compare -fopenmp
include ../kaldi.mk

BINFILES = arpa2fst arpa-to-const-arpa cued-rnnlm-eval

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
