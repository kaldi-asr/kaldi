
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = segmentation-copy segmentation-init-from-segments \
					 segmentation-init-from-ali segmentation-select-top \
					 gmm-acc-pdf-stats-segmentation select-feats-from-segmentation \
					 gmm-est-segmentation segmentation-to-rttm segmentation-post-process \
					 gmm-update-segmentation segmentation-to-ali

OBJFILES =



TESTFILES = 

ADDLIBS = ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a  ../segmenter/kaldi-segmenter.a \
					../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a ../tree/kaldi-tree.a

include ../makefiles/default_rules.mk

