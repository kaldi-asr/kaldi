
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = align-equal align-equal-compiled acc-tree-stats \
        show-alignments compile-questions cluster-phones \
        compute-wer make-h-transducer add-self-loops convert-ali \
        compile-train-graphs compile-train-graphs-fsts arpa2fst \
        make-pdf-to-tid-transducer make-ilabel-transducer show-transitions \
        ali-to-phones ali-to-post weight-silence-post acc-lda est-lda \
        ali-to-pdf est-mllt build-tree build-tree-two-level decode-faster \
        decode-faster-mapped vector-scale copy-transition-model rand-prune-post \
        phones-to-prons prons-to-wordali copy-gselect copy-tree scale-post \
        compute-mce-scale get-silence-probs post-to-weights reverse-weights \
        dot-weights sum-tree-stats weight-post post-to-tacc copy-matrix \
        copy-vector copy-int-vector sum-post sum-matrices draw-tree \
        copy-int-vector-vector thresh-post \
        align-mapped align-compiled-mapped latgen-faster-mapped latgen-faster-mapped-parallel \
        hmm-info pdf-to-counts analyze-counts extract-ctx post-to-phone-post \
        post-to-pdf-post duplicate-matrix logprob-to-post prob-to-post copy-post \
        matrix-logprob matrix-sum latgen-tracking-mapped \
        build-pfile-from-ali get-post-on-ali tree-info am-info \
        vector-sum matrix-sum-rows est-pca


OBJFILES =

ADDLIBS = ../lm/kaldi-lm.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
	      ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a \
          ../base/kaldi-base.a  ../thread/kaldi-thread.a


TESTFILES =

include ../makefiles/default_rules.mk

