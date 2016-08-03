EXTRA_CXXFLAGS = -Wno-sign-compare -lrt -fopenmp

all:

include ../kaldi.mk

#<<<<<<< HEAD
#TESTFILES = lm-lib-test
#
#OBJFILES = const-arpa-lm.o kaldi-lmtable.o kaldi-lm.o kaldi-rnnlm.o \
#           mikolov-rnnlm-lib.o cued-rnnlm-lib.o kaldi-cuedrnnlm.o
#=======
TESTFILES = arpa-file-parser-test arpa-lm-compiler-test
#>>>>>>> 6c7c0170812a1f7dfb5c09c078787e79ee72333a

OBJFILES = arpa-file-parser.o arpa-lm-compiler.o const-arpa-lm.o \
	   kaldi-rnnlm.o mikolov-rnnlm-lib.o kaldi-cuedrnnlm.o cued-rnnlm-lib.o

LIBNAME = kaldi-lm

ADDLIBS = ../fstext/kaldi-fstext.a ../util/kaldi-util.a \
          ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
