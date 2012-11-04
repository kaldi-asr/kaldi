
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-to-kws-index kws-index-union transcripts-to-fsts \
		   kws-search

OBJFILES = kws-functions.o

LIBFILE = kaldi-kws.a

all:  $(BINFILES)


TESTFILES =


$(BINFILES): $(LIBFILE) ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
  ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
  ../fstext/kaldi-fstext.a

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
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

