
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = fgmm-global-acc-stats fgmm-global-sum-accs fgmm-global-est \
           fgmm-global-merge fgmm-global-to-gmm fgmm-gselect fgmm-global-get-frame-likes \
           fgmm-global-acc-stats-twofeats fgmm-global-copy fgmm-global-mixdown \
           fgmm-global-gselect-to-post fgmm-global-info \
           fgmm-global-acc-stats-post fgmm-global-init-from-accs


OBJFILES =



TESTFILES =

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
