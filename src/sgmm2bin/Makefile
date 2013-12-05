
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = sgmm2-init sgmm2-gselect sgmm2-acc-stats sgmm2-est sgmm2-sum-accs \
         sgmm2-align-compiled sgmm2-est-spkvecs sgmm2-post-to-gpost \
         sgmm2-acc-stats-gpost sgmm2-latgen-faster sgmm2-est-spkvecs-gpost \
         sgmm2-rescore-lattice sgmm2-copy sgmm2-info sgmm2-est-ebw \
         sgmm2-acc-stats2 sgmm2-comp-prexform sgmm2-est-fmllr sgmm2-project \
         sgmm2-latgen-faster-parallel

OBJFILES =



TESTFILES =


ADDLIBS =  ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../feat/kaldi-feat.a \
	../sgmm2/kaldi-sgmm2.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
	../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
	../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
