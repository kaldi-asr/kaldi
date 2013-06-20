

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = nnet-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o nnet-cache.o nnet-cache-tgtmat.o \
	   nnet-loss-prior.o nnet-pdf-prior.o

LIBFILE = kaldi-nnet.a 

all:  $(LIBFILE)

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

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

