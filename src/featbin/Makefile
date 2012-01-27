
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-cmvn-stats add-deltas \
	remove-mean apply-cmvn transform-feats copy-feats compose-transforms \
    splice-feats extract-segments subset-feats feat-to-len feat-to-dim


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
	rm *.o *.a $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

