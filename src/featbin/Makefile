
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-fbank-feats \
    compute-cmvn-stats add-deltas remove-mean apply-cmvn transform-feats \
    copy-feats compose-transforms splice-feats extract-segments subset-feats \
    feat-to-len feat-to-dim fmpe-apply-transform fmpe-acc-stats fmpe-init \
    fmpe-est fmpe-copy fmpe-sum-accs append-feats extend-transform-dim \
    get-full-lda-mat compute-spectrogram-feats extract-feature-segments \
    reverse-feats paste-feats select-feats subsample-feats

OBJFILES = 

all:  $(BINFILES)


TESTFILES =

$(BINFILES): ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
         ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
         ../base/kaldi-base.a

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

