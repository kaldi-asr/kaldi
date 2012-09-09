
all:

include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare

TESTFILES = kaldi-lattice-test

OBJFILES = kaldi-lattice.o lattice-functions.o word-align-lattice.o sausages.o

LIBFILE = kaldi-lat.a

all:  $(LIBFILE)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../util/kaldi-util.a ../base/kaldi-base.a


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

