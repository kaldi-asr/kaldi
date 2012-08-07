
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = sgmm2-init sgmm2-gselect sgmm2-acc-stats sgmm2-est sgmm2-sum-accs \
         sgmm2-align-compiled sgmm2-est-spkvecs sgmm2-post-to-gpost \
         sgmm2-acc-stats-gpost sgmm2-latgen-faster sgmm2-est-spkvecs-gpost \
         sgmm2-rescore-lattice sgmm2-copy sgmm2-info sgmm2-est-ebw \
         sgmm2-acc-stats2 sgmm2-comp-prexform sgmm2-est-fmllr sgmm2-project

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../sgmm2/kaldi-sgmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

