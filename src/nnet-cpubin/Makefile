
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

BINFILES = nnet-randomize-frames nnet-am-info nnet-train nnet-init \
   nnet-train-simple nnet-train-transitions nnet-latgen-faster nnet-am-copy \
   nnet-add-hidden-layer nnet-am-init nnet-insert nnet-align-compiled \
   nnet-compute-prob nnet-copy-egs nnet-shrink nnet-combine nnet-combine-a \
   nnet-am-average nnet-am-combine nnet-am-compute

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../nnet-cpu/kaldi-nnet-cpu.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a 



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

