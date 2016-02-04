EXTRA_CXXFLAGS = -Wno-sign-compare

all:

include ../kaldi.mk

TESTFILES = arpa-file-parser-test arpa-lm-compiler-test

OBJFILES = arpa-file-parser.o arpa-lm-compiler.o const-arpa-lm.o \
	   kaldi-rnnlm.o mikolov-rnnlm-lib.o

LIBNAME = kaldi-lm

ADDLIBS = ../base/kaldi-base.a ../fstext/kaldi-fstext.a ../util/kaldi-util.a

include ../makefiles/default_rules.mk
