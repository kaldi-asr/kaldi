
all:

include ../kaldi.mk

TESTFILES = kaldi-thread-test kaldi-task-sequence-test

OBJFILES =  kaldi-thread.o kaldi-mutex.o kaldi-semaphore.o kaldi-barrier.o

LIBNAME = kaldi-thread
ADDLIBS = ../base/kaldi-base.a 


include ../makefiles/default_rules.mk
