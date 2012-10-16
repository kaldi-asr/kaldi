

all:

include ../kaldi.mk

TESTFILES = feature-mfcc-test feature-plp-test feature-fbank-test

OBJFILES = feature-functions.o feature-mfcc.o feature-plp.o feature-fbank.o \
              feature-spectrogram.o mel-computations.o wave-reader.o 

LIBFILE = kaldi-feature.a 

all:  $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)



$(TESTFILES): $(LIBFILE) ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../util/kaldi-util.a \
	../matrix/kaldi-matrix.a ../base/kaldi-base.a

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

