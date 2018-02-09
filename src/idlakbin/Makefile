
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-aperiodic-feats compute-minmax-stats apply-minmax \
           ali-to-hmmstate make-fullctx-ali make-fullctx-ali-dnn

OBJFILES =

TESTFILES =

ADDLIBS = ../idlakfeat/idlak-feat.a ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../hmm/kaldi-hmm.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a \

include ../makefiles/default_rules.mk

