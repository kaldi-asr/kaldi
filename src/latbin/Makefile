
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-best-path lattice-prune lattice-equivalent lattice-to-nbest \
           lattice-lmrescore lattice-scale lattice-union lattice-to-post \
           lattice-determinize lattice-oracle lattice-rmali \
           lattice-compose lattice-boost-ali lattice-copy lattice-to-fst \
           lattice-to-phone-lattice lattice-interp lattice-project \
           lattice-add-trans-probs lattice-difference lattice-word-align \
           nbest-to-linear nbest-to-lattice lattice-1best linear-to-nbest \
           lattice-mbr-decode lattice-align-words lattice-to-mpe-post \
           lattice-copy-backoff nbest-to-ctm


OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
	../matrix/kaldi-matrix.a ../base/kaldi-base.a 



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

