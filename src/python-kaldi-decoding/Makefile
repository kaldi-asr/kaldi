# We suppose that this Makefile sits in kaldi-trunk/src/ThisDirectory
all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I ../../tools/portaudio/install/include
EXTRA_LDLIBS =

include ../kaldi.mk
CC = gcc

TESTFILES = compute-wer-test gmm-latgen-faster-test compute-mfcc-feats-test \
			lattice-best-path-test online-wav-gmm-decode-faster-test
OBJFILES = compute-wer.o gmm-latgen-faster.o compute-mfcc-feats.o \
		   lattice-best-path.o online-wav-gmm-decode-faster.o 

LDLIBFILE = libkaldi-cffi.so
LIBFILE = kaldi-cffi.a


UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
    EXTRA_LDLIBS += ../../tools/portaudio/install/lib/libportaudio.a
ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
else
    EXTRA_LDLIBS += -lrt
endif
else
    EXTRA_LDLIBS += -L $(PA_LDD) -lportaudio
endif

all:  $(LIBFILE) $(LDLIBFILE)

# I can not use ../decoder/kaldi-decoder.a because there are two types of decoders for two types sgmm vs sgmm2
$(LDLIBFILE): $(OBJFILES) \
	../decoder/decodable-am-diag-gmm.o ../decoder/lattice-faster-decoder.o ../decoder/faster-decoder.o \
	../online/kaldi-online.a ../thread/kaldi-thread.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
	../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../fstext/kaldi-fstext.a ../tree/kaldi-tree.a \
	../matrix/kaldi-matrix.a ../feat/kaldi-feature.a ../util/kaldi-util.a ../base/kaldi-base.a
	$(CC) -fPIC -shared -o $@ -Wl,-export-dynamic,--whole-archive $^ -Wl,--no-whole-archive $(LDLIBS)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
		-rm -f *.o *.a tmp* *.tmp *.so .depend.mk $(TESTFILES) valgrind.out

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
# .depend.mk: depend

-include .depend.mk

.valgrind: $(TESTFILES)

.PHONY: test_dyn python_test test

### TESTS - launching command --help ### 
test_dyn: $(LDLIBFILE) $(TESTFILES)
	# Before running the binaries set the LD_LIBRARY_PATH variable as we do!
	@result=0; for x in $(TESTFILES); do \
		echo -n "Running $$x :  "; \
		LD_LIBRARY_PATH=$(OBLAS_LDD):$(FST_LDD):$:`pwd` ./$$x --help  > /dev/null 2>&1; \
		if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  \
		done; exit $$result

python_test: run.py $(LDLIBFILE) 
	LD_LIBRARY_PATH=$(OBLAS_LDD):$(FST_LDD):`pwd` python $<
	echo $?

# test: test_dyn python_test
test: test_dyn
