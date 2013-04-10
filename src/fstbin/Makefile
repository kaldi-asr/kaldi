
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = 

OBJFILES = 

BINFILES = fstdeterminizestar  \
           fstrmsymbols fstisstochastic fstminimizeencoded fstmakecontextfst \
           fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops  \
           fstrmepslocal fstcomposecontext fsttablecompose fstrand fstfactor \
           fstdeterminizelog fstphicompose fstrhocompose fstpropfinal fstcopy \
	       fstpushspecial fsts-to-transcripts

# actually, this library is currently empty.  Everything is a header.
LIBFILE = 

all: $(BINFILES)


$(BINFILES): ../fstext/kaldi-fstext.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../util/kaldi-util.a 

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

