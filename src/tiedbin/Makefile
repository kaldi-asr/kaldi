
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = \
  tied-diag-gmm-init-mono \
  tied-diag-gmm-acc-stats-ali \
  tied-diag-gmm-est \
  tied-diag-gmm-align-compiled \
  tied-diag-gmm-decode-simple \
  tied-diag-gmm-decode-faster \
  tied-diag-gmm-init-model \
  smooth-stats-diag \
  tied-full-gmm-init-mono \
  tied-full-gmm-acc-stats-ali \
  tied-full-gmm-est \
  tied-full-gmm-align-compiled \
  tied-full-gmm-decode-simple \
  tied-full-gmm-decode-faster \
  tied-full-gmm-init-model \
  smooth-stats-full \
  init-tied-codebooks \
  tied-lbg \
  full-to-diag

OBJFILES =



TESTFILES =


ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../feat/kaldi-feat.a \
	../transform/kaldi-transform.a ../tied/kaldi-tied.a \
	../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
	../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
