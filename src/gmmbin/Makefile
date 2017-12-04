
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = gmm-init-mono gmm-est gmm-acc-stats-ali gmm-align \
           gmm-decode-faster gmm-decode-simple gmm-decode-nbest gmm-align-compiled \
           gmm-sum-accs gmm-init-trans gmm-est-regtree-fmllr gmm-acc-stats-twofeats \
           gmm-acc-stats gmm-init-lvtln gmm-est-lvtln-trans gmm-train-lvtln-special \
           gmm-acc-mllt gmm-mixup gmm-init-model \
           gmm-acc-hlda gmm-est-hlda gmm-transform-means \
           gmm-make-regtree gmm-decode-faster-regtree-fmllr gmm-post-to-gpost \
           gmm-est-fmllr-gpost gmm-est-fmllr gmm-est-regtree-fmllr-ali \
           gmm-est-regtree-mllr gmm-compute-likes \
           gmm-decode-faster-regtree-mllr gmm-latgen-simple \
           gmm-rescore-lattice gmm-decode-biglm-faster \
           gmm-est-gaussians-ebw gmm-est-weights-ebw gmm-latgen-faster gmm-copy \
           gmm-global-acc-stats gmm-global-est gmm-global-sum-accs gmm-gselect \
           gmm-latgen-biglm-faster gmm-ismooth-stats gmm-global-get-frame-likes \
           gmm-global-est-fmllr gmm-global-to-fgmm gmm-global-acc-stats-twofeats \
           gmm-global-copy gmm-get-feat-deriv \
           gmm-fmpe-acc-stats gmm-acc-stats2 gmm-init-model-flat gmm-info \
           gmm-scale-accs gmm-get-stats-deriv gmm-est-rescale gmm-boost-silence \
           gmm-diff-accs gmm-basis-fmllr-accs gmm-basis-fmllr-training gmm-est-basis-fmllr \
           gmm-est-map gmm-adapt-map gmm-latgen-map gmm-basis-fmllr-accs-gpost \
           gmm-est-basis-fmllr-gpost gmm-latgen-faster-parallel \
           gmm-est-fmllr-raw gmm-est-fmllr-raw-gpost gmm-global-init-from-feats \
           gmm-global-info gmm-latgen-faster-regtree-fmllr gmm-est-fmllr-global \
           gmm-acc-mllt-global gmm-transform-means-global gmm-global-get-post \
           gmm-global-gselect-to-post gmm-global-est-lvtln-trans gmm-init-biphone

OBJFILES =



TESTFILES =

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
