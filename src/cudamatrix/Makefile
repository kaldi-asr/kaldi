

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = cuda-matrix-test 

OBJFILES = cu-device.o 
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o cu-randkernels.o
endif

LIBFILE = cuda-matrix.a

all:  $(LIBFILE)

#implicit rule for kernel compilation
#symlink in necessary for cross-compilation at BUT
%.o : %.cu
	-ln -s `which $(CC)` gcc
	export PATH=$(CUDATKDIR)/bin:$(PWD):$(PWD)/cudamatrix:$(PATH); ln -s `which $(CC)` gcc; $(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) -I../
	-unlink gcc


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 


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

