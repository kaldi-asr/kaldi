all:

include ../kaldi.mk

TESTFILES = regtree-fmllr-diag-gmm-test lda-estimate-test \
      regression-tree-test fmllr-diag-gmm-test exponential-transform-test \
      regtree-mllr-diag-gmm-test fmpe-test

OBJFILES = regression-tree.o regtree-mllr-diag-gmm.o lda-estimate.o \
    regtree-fmllr-diag-gmm.o cmvn.o transform-common.o fmllr-diag-gmm.o \
    lvtln.o mllt.o exponential-transform.o hlda.o fmpe.o basis-fmllr-diag-gmm.o \
    compressed-transform-stats.o


LIBFILE = kaldi-transform.a

all:  $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) tmp* *.tmp

test_compile: $(TESTFILES)

test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

