
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = language-model-test cctc-transition-model-test \
		    cctc-supervision-test

OBJFILES = language-model.o cctc-transition-model.o cctc-graph.o \
           cctc-supervision.o cctc-training.o cctc-test-utils.o

LIBNAME = kaldi-ctc

ADDLIBS = ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../lat/kaldi-lat.a ../cudamatrix/kaldi-cudamatrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk

