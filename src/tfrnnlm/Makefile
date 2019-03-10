# This directory is not compiled automatically when you do make under src/
# to compile the TensorFlow related code,

# you would need to run tools/extras/install_tensorflow_cc.sh first to generate
# the static library files required for compilation, and then

# cd ${KALDI_ROOT}/src
# cd tfrnnlm
# make
# cd ../tfrnnlmbin/
# make

include ../kaldi.mk

TENSORFLOW = ../../tools/tensorflow

all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I$(TENSORFLOW)/bazel-tensorflow/external/protobuf/src \
                 -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/nsync/public \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

OBJFILES = tensorflow-rnnlm.o

TESTFILES =

LIBNAME = kaldi-tensorflow-rnnlm

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 
LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += -L$(TENSORFLOW)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework

include ../makefiles/default_rules.mk
