all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I ../../tools/portaudio/install/include

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
else
    EXTRA_LDLIBS += -lrt
endif
else
    EXTRA_LDLIBS = -L ../../tools/portaudio/install/lib/ -lportaudio
endif

include ../kaldi.mk

BINFILES = online-net-client online-server-gmm-decode-faster online-gmm-decode-faster online-wav-gmm-decode-faster

OBJFILES =

LIBFILE =

all:  $(LIBFILE) $(BINFILES)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(BINFILES): $(LIBFILE) ../online/kaldi-online.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(BINFILES) tmp* *.tmp

#test: $(BINFILES)
#	for x in $(BINFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done
#	echo Tests succeeded

.valgrind: $(BINFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

