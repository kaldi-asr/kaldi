EXTRA_CXXFLAGS = -Wno-sign-compare

all:

include ../kaldi.mk

TESTFILES = arpa-file-parser-test arpa-lm-compiler-test

TESTFILES += faster-arpa-lm-test

OBJFILES = arpa-file-parser.o arpa-lm-compiler.o const-arpa-lm.o \
	   kaldi-rnnlm.o mikolov-rnnlm-lib.o  faster-arpa-lm.o 

LIBNAME = kaldi-lm

ADDLIBS = ../fstext/kaldi-fstext.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
