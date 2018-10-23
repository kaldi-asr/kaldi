
all:

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = sampler-test sampling-lm-test rnnlm-example-test

OBJFILES = sampler.o rnnlm-example.o rnnlm-example-utils.o \
           rnnlm-core-training.o rnnlm-embedding-training.o rnnlm-core-compute.o \
           rnnlm-utils.o rnnlm-training.o rnnlm-test-utils.o sampling-lm-estimate.o \
           sampling-lm.o rnnlm-compute-state.o rnnlm-lattice-rescoring.o

LIBNAME = kaldi-rnnlm

ADDLIBS = ../nnet3/kaldi-nnet3.a ../cudamatrix/kaldi-cudamatrix.a \
          ../lm/kaldi-lm.a ../hmm/kaldi-hmm.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
