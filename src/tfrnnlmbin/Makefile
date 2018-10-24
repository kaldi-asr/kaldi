# This directory is not compiled automatically when you do make under src/
# to compile the TensorFlow related code, do the following:

# you would need to run tools/extras/install_tensorflow_cc.sh first to generate
# the static library files required for compilation, and then

# cd ${KALDI_ROOT}/src
# cd tfrnnlm
# make
# cd ../tfrnnlmbin/
# make

TENSORFLOW = $(shell pwd)/../../tools/tensorflow

all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src \
                 -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/nsync/public \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src
include ../kaldi.mk

BINFILES = lattice-lmrescore-tf-rnnlm lattice-lmrescore-tf-rnnlm-pruned

OBJFILES =

TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
          ../tfrnnlm/kaldi-tensorflow-rnnlm.a 

LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework

LDFLAGS += -Wl,-rpath=$(shell pwd)/../../tools/tensorflow/bazel-bin/tensorflow/

include ../makefiles/default_rules.mk
