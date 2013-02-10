

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = cuda-matrix-test 

OBJFILES = cu-device.o cu-math.o cu-matrix.o
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o cu-randkernels.o
endif

LIBFILE = cuda-matrix.a

all:  $(LIBFILE)


ifeq ($(CUDA), true)
  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_10,code=sm_10 \
	    -gencode arch=compute_13,code=sm_13 \
	    -gencode arch=compute_20,code=sm_20
  #Get the CUDA Toolkit version (remove decimal point char)
  CUDA_VERSION=$(shell $(CUDATKDIR)/bin/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
  #For toolkit 4.2 or newer, add the compute capability 3.0 
  CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
  ifeq ($(CUDA_VER_GT_4_2), true)
    CUDA_ARCH += -gencode arch=compute_30,code=sm_30
  endif
endif


#implicit rule for kernel compilation
#symlink in necessary for cross-compilation at BUT
%.o : %.cu
	-ln -s `which $(CC)` gcc
	export PATH=$(CUDATKDIR)/bin:$(PWD):$(PWD)/cudamatrix:$(PATH); $(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../
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

