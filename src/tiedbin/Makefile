
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = tied-diag-gmm-init-mono tied-diag-gmm-acc-stats-ali \
           tied-diag-gmm-est tied-diag-gmm-align-compiled \
           tied-diag-gmm-decode-simple tied-diag-gmm-decode-faster \
           tied-full-gmm-init-mono tied-full-gmm-acc-stats-ali \
           tied-full-gmm-est tied-full-gmm-align-compiled \
           tied-full-gmm-decode-simple tied-full-gmm-decode-faster \
           full-to-tied tied-diag-gmm-init-model tied-diag-gmm-align \
           tied-full-gmm-align

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../tied/kaldi-tied-gmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 



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

