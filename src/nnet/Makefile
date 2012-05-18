

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = nnet-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o nnet-cache.o nnet-cache-tgtmat.o

LIBFILE = kaldi-nnet.a 

all:  $(LIBFILE) $(TESTFILES)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)



$(TESTFILES): $(LIBFILE) ../cudamatrix/cuda-matrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

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

