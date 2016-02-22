
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-vad select-voiced-frames \
           create-split-from-vad \
					 select-top-frames select-top-chunks \
					 select-interior-frames vector-extract-dims

OBJFILES =



TESTFILES =


ADDLIBS = ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
    ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
