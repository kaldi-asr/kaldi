
all:

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = sampler-test arpa-sampling-test

OBJFILES = sampler.o arpa-sampling.o rnnlm-example.o rnnlm-example-utils.o \
           rnnlm-core-training.o

LIBNAME = kaldi-rnnlm

ADDLIBS = ../nnet3/kaldi-nnet3.a ../cudamatrix/kaldi-cudamatrix.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a \
          ../lm/kaldi-lm.a

include ../makefiles/default_rules.mk

