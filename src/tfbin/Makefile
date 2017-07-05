
CURDIR = $(shell pwd)
TENSORFLOW = $(CURDIR)/../../tools/tensorflow

all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen/
include ../kaldi.mk

BINFILES = lattice-lmrescore-tf-rnnlm

OBJFILES =

CURDIR = $(shell pwd)

TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../tensorflow/kaldi-tensorflow-rnnlm.a \
          $(CURDIR)/../../tools/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so

LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += $(OTHERLIBS) -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc

include ../makefiles/default_rules.mk
