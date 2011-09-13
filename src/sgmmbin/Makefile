
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = init-ubm sgmm-align sgmm-align-compiled sgmm-acc-stats-ali \
           sgmm-sum-accs sgmm-est sgmm-decode-faster sgmm-init sgmm-gselect \
           sgmm-acc-stats sgmm-est-spkvecs sgmm-post-to-gpost \
           sgmm-acc-stats-gpost sgmm-est-spkvecs-gpost sgmm-comp-prexform \
           sgmm-est-fmllr-gpost sgmm-acc-fmllrbasis-ali sgmm-est-fmllrbasis \
           sgmm-calc-distances sgmm-normalize sgmm-latgen-simple \
           sgmm-rescore-lattice

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../sgmm/kaldi-sgmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 



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

