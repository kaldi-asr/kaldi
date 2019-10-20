
# make "all" the target.
all:

##$ c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `` example.cpp -o example`python3-config --extension-suffix`



# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk


ifeq ($(KALDI_FLAVOR),static)
  $(error You cannot build the pybind directory unless the build was dynamic; reconfigure with --shared option.)
endif

PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes)
CXXFLAGS += $(PYBIND_INCLUDES)
PYBIND_EXTENSION=$(shell python3-config --extension-suffix)
LIBFILE=kaldi-pybind
LIBFILE_EXTENSION=$(PYBIND_EXTENSION)

CXXFLAGS += -O3

ifeq ($(shell uname),Darwin)
  LDFLAGS += -undefined dynamic_lookup
endif

OBJFILES = kaldi_pybind.o

LIBNAME = kaldi_pybind

# Note: the following will actually get turned into dynamic library names.
ADDLIBS = ../lib/kaldi-util.so ../lib/kaldi-matrix.so ../lib/kaldi-base.so

include ../makefiles/default_rules.mk
