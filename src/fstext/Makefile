
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = determinize-star-test \
      pre-determinize-test trivial-factor-weight-test \
      context-fst-test factor-test table-matcher-test fstext-utils-test \
      remove-eps-local-test rescale-test lattice-weight-test  \
      determinize-lattice-test lattice-utils-test deterministic-fst-test \
      determinize-lattice-pruned-test

OBJFILES = push-special.o


LIBFILE = kaldi-fstext.a

all: $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)

# tree and matrix archives needed for test-context-fst
# matrix archive needed for push-special.
$(TESTFILES): $(LIBFILE)  ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a 

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

