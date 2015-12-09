
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-fbank-feats \
    compute-cmvn-stats add-deltas remove-mean apply-cmvn transform-feats \
    copy-feats compose-transforms splice-feats extract-segments subset-feats \
    feat-to-len feat-to-dim fmpe-apply-transform fmpe-acc-stats fmpe-init \
    fmpe-est fmpe-copy fmpe-sum-accs extend-transform-dim \
    get-full-lda-mat compute-spectrogram-feats extract-feature-segments \
    reverse-feats paste-feats select-feats subsample-feats process-pitch-feats \
    interpolate-pitch copy-feats-to-htk copy-feats-to-sphinx extract-rows \
    apply-cmvn-sliding compute-cmvn-stats-two-channel compute-kaldi-pitch-feats \
    process-kaldi-pitch-feats compare-feats wav-to-duration add-deltas-sdc \
    compute-and-process-kaldi-pitch-feats modify-cmvn-stats wav-copy \
    wav-reverberate append-vector-to-feats detect-sinusoids shift-feats \
    concat-feats

OBJFILES =

TESTFILES =

ADDLIBS = ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
         ../thread/kaldi-thread.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
         ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

