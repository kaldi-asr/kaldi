include ../kaldi.mk

CURDIR = $(shell pwd)
TENSORFLOW = $(CURDIR)/../../tools/tensorflow

all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen/
#EXTRA_CXXFLAGS = -Wno-sign-compare -fPIC -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen/

OBJFILES = tensorflow-rnnlm-lib.o

TESTFILES =

LIBNAME = kaldi-tensorflow-rnnlm

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
          $(CURDIR)/../../tools/tensorflow/bazel-bin/tensorflow/tensorflow_cc.so

LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += $(OTHERLIBS) -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc

include ../makefiles/default_rules.mk
