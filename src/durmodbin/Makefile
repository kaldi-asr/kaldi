
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


BINFILES = nnet3-durmodel-init nnet3-durmodel-make-egs nnet3-durmodel-copy \
           nnet3-durmodel-rescore-lattice durmodel-info durmodel-init \
           nnet3-durmodel-info nnet3-durmodel-copy-egs \
           nnet3-durmodel-estimate-avg-logprobs durmodel-context-stats \
           nnet3-durmodel-find-avg-objf-factor nnet3-durmodel-plot-outputs \
           nnet3-durmodel-investigate
OBJFILES =



TESTFILES =


ADDLIBS = ../durmod/kaldi-durmod.a ../nnet3/kaldi-nnet3.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a \
          ../thread/kaldi-thread.a \
          ../fstext/kaldi-fstext.a ../cudamatrix/kaldi-cudamatrix.a

include ../makefiles/default_rules.mk
