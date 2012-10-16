
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
        decode-faster-mapped scale-vecs copy-transition-model rand-prune-post \
        phones-to-prons prons-to-wordali copy-gselect copy-tree scale-post \
        compute-mce-scale get-silence-probs post-to-weights reverse-weights \
        dot-weights sum-tree-stats weight-post post-to-tacc copy-matrix \
        copy-vector copy-int-vector sum-post sum-matrices draw-tree \
        copy-int-vector-vector thresh-post \
        align-mapped align-compiled-mapped latgen-faster-mapped \
        hmm-info analyze-counts

OBJFILES = 

all:  $(BINFILES)


TESTFILES =

$(BINFILES): ../lm/kaldi-lm.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES) 

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

