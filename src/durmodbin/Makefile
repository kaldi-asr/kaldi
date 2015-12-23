
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = durmod-init durmod-make-egs durmod-copy-nnet
OBJFILES =



TESTFILES =


ADDLIBS = ../durmod/kaldi-durmod.a ../nnet3/kaldi-nnet3.a \
					../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
					../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
					../util/kaldi-util.a ../base/kaldi-base.a \
					../cudamatrix/kaldi-cudamatrix.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk
