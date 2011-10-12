
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = gmm-init-mono gmm-est gmm-acc-stats-ali gmm-align \
           gmm-decode-faster gmm-decode-simple gmm-decode-nbest gmm-align-compiled \
           gmm-sum-accs gmm-init-trans gmm-est-regtree-fmllr gmm-acc-stats-twofeats \
           gmm-acc-stats gmm-init-lvtln gmm-est-lvtln-trans gmm-train-lvtln-special \
           gmm-acc-mllt gmm-mixup gmm-init-model \
           gmm-acc-hlda gmm-est-hlda gmm-transform-means gmm-init-et gmm-est-et \
           gmm-et-acc-a gmm-et-est-a gmm-copy-et gmm-et-get-b \
           gmm-make-regtree gmm-decode-faster-regtree-fmllr gmm-post-to-gpost \
           gmm-est-fmllr-gpost gmm-est-fmllr gmm-est-regtree-fmllr-ali \
           gmm-est-regtree-mllr gmm-decode-kaldi gmm-compute-likes \
           gmm-decode-faster-regtree-mllr gmm-et-apply-c gmm-latgen-simple \
           gmm-rescore-lattice gmm-decode-biglm-faster fmpe-gmm-model-diffs-est \
           fmpe-gmm-acc-stats-gpost fmpe-gmm-sum-accs fmpe-init-gmms fmpe-gmm-est \
           gmm-est-mmi gmm-latgen-faster

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >&/dev/null || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

