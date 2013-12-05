
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = init-ubm sgmm-align-compiled sgmm-acc-stats-ali \
           sgmm-sum-accs sgmm-est sgmm-decode-faster sgmm-init sgmm-gselect \
           sgmm-est-fmllr sgmm-acc-stats sgmm-est-spkvecs sgmm-post-to-gpost \
           sgmm-acc-stats-gpost sgmm-est-spkvecs-gpost sgmm-comp-prexform \
           sgmm-est-fmllr-gpost sgmm-acc-fmllrbasis-ali sgmm-est-fmllrbasis \
           sgmm-calc-distances sgmm-normalize sgmm-latgen-simple \
           sgmm-latgen-faster sgmm-rescore-lattice sgmm-copy sgmm-write-ubm \
           sgmm-mixup sgmm-info sgmm-acc-tree-stats sgmm-sum-tree-stats \
           sgmm-build-tree sgmm-cluster-phones sgmm-init-from-tree-stats \
           sgmm-est-ebw sgmm-acc-stats2 sgmm-est-multi

OBJFILES =



TESTFILES =


ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a  ../feat/kaldi-feat.a \
	../sgmm/kaldi-sgmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
	../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
	../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
