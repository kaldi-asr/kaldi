
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = compose-trim-test determinize-star-test make-stochastic-test pre-determinize-test trivial-factor-weight-test \
      context-fst-test factor-test table-matcher-test fstext-utils-test \
      remove-eps-local-test rescale-test reorder-test

OBJFILES = 

# actually, this library is currently empty.  Everything is a header.
LIBFILE = 

all: $(LIBFILE) $(TESTFILES)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)

# tree and matrix archives needed for test-context-fst
$(TESTFILES): $(LIBFILE)  ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../util/kaldi-util.a

%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >&/dev/null || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

