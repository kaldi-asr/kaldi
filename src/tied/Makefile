all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = am-tied-diag-gmm-test am-tied-full-gmm-test

OBJFILES = tied-gmm.o mle-tied-gmm.o am-tied-diag-gmm.o mle-am-tied-diag-gmm.o am-tied-diag-gmm-test.o am-tied-full-gmm.o mle-am-tied-full-gmm.o

LIBFILE = kaldi-tied-gmm.a

all:  $(LIBFILE)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../base/kaldi-base.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
	../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a


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

