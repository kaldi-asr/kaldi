# Whenever make is run in this directory, call ./get_version.sh script.
# ./get_version.sh script regenerates ./version.h if necessary, e.g. if it
# does not already exist or if the version number has changed.
RET := $(shell ./get_version.sh; echo " $$?")
ifneq ($(strip $(RET)), 0)
  RC := $(lastword $(RET))
  OUT := $(wordlist 1,$(shell echo $$(($(words $(RET))-1))),$(RET))
  ifeq ($(RC),0)
    $(info $(OUT))
  else
    $(error $(OUT))
  endif
endif

all:

include ../kaldi.mk

TESTFILES = kaldi-math-test io-funcs-test kaldi-error-test timer-test

OBJFILES = kaldi-math.o kaldi-error.o io-funcs.o kaldi-utils.o

LIBNAME = kaldi-base

ADDLIBS =

include ../makefiles/default_rules.mk
