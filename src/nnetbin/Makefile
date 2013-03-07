
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

BINFILES = nnet-train-xent-hardlab-perutt nnet-train-xent-hardlab-frmshuff \
	nnet-train-mse-tgtmat-frmshuff nnet-forward nnet-copy rbm-train-cd1-frmshuff \
	rbm-convert-to-nnet nnet-trim-n-last-transforms nnet-train-mmi-sequential \
	nnet-train-xent-hardlab-frmshuff-prior transf-to-nnet cmvn-to-nnet \
	nnet-mpe

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../nnet/kaldi-nnet.a ../cudamatrix/cuda-matrix.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o $(TESTFILES) $(BINFILES)

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

