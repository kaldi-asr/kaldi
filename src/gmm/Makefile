all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = diag-gmm-test mle-diag-gmm-test full-gmm-test mle-full-gmm-test \
		am-diag-gmm-test mle-am-diag-gmm-test ebw-diag-gmm-test

OBJFILES = diag-gmm.o diag-gmm-normal.o mle-diag-gmm.o am-diag-gmm.o mle-am-diag-gmm.o \
		full-gmm.o full-gmm-normal.o mle-full-gmm.o model-common.o \
		model-test-common.o ebw-diag-gmm.o indirect-diff-diag-gmm.o \
        map-diag-gmm-accs.o

LIBFILE = kaldi-gmm.a

all:  $(LIBFILE)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) tmp* *.tmp

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || { echo "***test $$x failed***"; exit 1; }; done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

