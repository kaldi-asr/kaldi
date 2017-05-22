
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = segmentation-copy segmentation-get-stats \
					 segmentation-init-from-ali segmentation-to-ali \
					 segmentation-init-from-segments segmentation-to-segments \
					 segmentation-combine-segments segmentation-merge-recordings \
					 segmentation-create-subsegments segmentation-intersect-ali \
					 segmentation-to-rttm segmentation-post-process \
					 segmentation-merge segmentation-split-segments \
					 segmentation-remove-segments \
					 segmentation-init-from-lengths \
					 segmentation-combine-segments-to-recordings \
					 segmentation-create-overlapped-subsegments \
					 segmentation-intersect-segments \
					 segmentation-init-from-additive-signals-info \
					 class-counts-per-frame-to-labels \
					 segmentation-speed-perturb \
					 intersect-int-vectors 

OBJFILES =



TESTFILES = 

ADDLIBS = ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a  ../segmenter/kaldi-segmenter.a ../tree/kaldi-tree.a \
					../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

