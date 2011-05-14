
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = 

OBJFILES = 

BINFILES = fstgetnextsymbol fstmakedisambigsyms fstpredeterminize fstdeterminizestar  \
           fstrmsymbols fstisstochastic fstminimizeencoded fstoptimize fstmakecontextfst \
           fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops  \
           fstrmepslocal fstcomposecontext fsttablecompose fstrand fstfactor \
           fstdeterminizelog fstreorder fstremoveuselessarcs

# actually, this library is currently empty.  Everything is a header.
LIBFILE = 

all: $(BINFILES)


$(BINFILES): ../base/kaldi-base.a ../util/kaldi-util.a

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

