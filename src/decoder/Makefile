all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../kaldi.mk

# Removing kaldi-decoder-test because it doesn't work without access
# to certain files...
#TESTFILES =  kaldi-decoder-test
TESTFILES = 

OBJFILES = decodable-am-diag-gmm.o training-graph-compiler.o decodable-am-sgmm.o decodable-am-tied-diag-gmm.o decodable-am-tied-full-gmm.o

LIBFILE = kaldi-decoder.a


all:  $(LIBFILE) $(TESTFILES)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
     ../sgmm/kaldi-sgmm.a ../gmm/kaldi-gmm.a ../util/kaldi-util.a \
     ../base/kaldi-base.a ../matrix/kaldi-matrix.a

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
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

