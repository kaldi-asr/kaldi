
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = add-deltas add-deltas-sdc append-post-to-feats \
           append-vector-to-feats apply-cmvn apply-cmvn-sliding compare-feats \
           compose-transforms compute-and-process-kaldi-pitch-feats \
           compute-cmvn-stats compute-cmvn-stats-two-channel \
           compute-fbank-feats compute-kaldi-pitch-feats compute-mfcc-feats \
           compute-plp-feats compute-spectrogram-feats concat-feats copy-feats \
           copy-feats-to-htk copy-feats-to-sphinx extend-transform-dim \
           extract-feature-segments extract-segments feat-to-dim \
           feat-to-len fmpe-acc-stats fmpe-apply-transform fmpe-est \
           fmpe-init fmpe-sum-accs get-full-lda-mat interpolate-pitch \
           modify-cmvn-stats paste-feats post-to-feats \
           process-kaldi-pitch-feats process-pitch-feats \
           select-feats shift-feats splice-feats subsample-feats \
           subset-feats transform-feats wav-copy wav-reverberate \
           wav-to-duration multiply-vectors paste-vectors 

OBJFILES =

TESTFILES =

ADDLIBS = ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
