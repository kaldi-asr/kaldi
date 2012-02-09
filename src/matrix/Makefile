

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =

include ../kaldi.mk


TESTFILES = matrix-lib-test

OBJFILES = kaldi-matrix.o kaldi-vector.o packed-matrix.o sp-matrix.o tp-matrix.o \
           matrix-functions.o srfft.o

LIBFILE = kaldi-matrix.a

all:  $(LIBFILE) $(TESTFILES)

$(LIBFILE): $(OBJFILES)
	while ! mkdir .lock; do sleep 1; done
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)
	rmdir .lock


$(TESTFILES): $(LIBFILE) ../base/kaldi-base.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

