all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I ../../tools/portaudio/install/include
include ../kaldi.mk

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
ifeq ($(KALDI_FLAVOR), dynamic)
		#The PA_RingBuffer interface is internal and is not exported in the so libray
		#so we have to compile in the static one
		#.PA_ABS_PATH := $(shell readlink -f ../../tools/portaudio/install/lib )
		#EXTRA_LDLIBS = -L ../../tools/portaudio/install/lib/ -Wl,-rpath=$(.PA_ABS_PATH) -lportaudio
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
else
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
endif
ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
else
    EXTRA_LDLIBS += -lrt
endif
else
    EXTRA_LDLIBS = -L ../../tools/portaudio/install/lib/ -lportaudio
endif


BINFILES = online-net-client online-server-gmm-decode-faster online-gmm-decode-faster \
           online-wav-gmm-decode-faster online-audio-server-decode-faster \
					 online-audio-client

OBJFILES =


TESTFILES =




ADDLIBS = ../online/kaldi-online.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
					../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
					../util/kaldi-util.a ../base/kaldi-base.a 

ifeq ($(KALDI_FLAVOR), dynamic)
ifdef LIBNAME
LIBFILE = lib$(LIBNAME).so
LDLIBS  += -l$(LIBNAME)
endif
LDFLAGS += $(foreach dep,$(ADDLIBS), -L$(dir $(dep)) )
LDFLAGS += -L.
LDFLAGS += -Wl,-rpath=$(shell readlink -f $(KALDILIBDIR))
LDLIBS  += $(foreach dep,$(ADDLIBS), -l$(notdir $(basename $(dep))) )
XDEPENDS = $(foreach dep,$(ADDLIBS), $(dir $(dep))/lib$(notdir $(basename $(dep))).so )
else
ifdef LIBNAME
LIBFILE = $(LIBNAME).a
endif
XDEPENDS = $(ADDLIBS)
endif

all: $(LIBFILE) $(BINFILES)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBNAME).a $(OBJFILES)
	$(RANLIB) $(LIBNAME).a
ifeq ($(KALDI_FLAVOR), dynamic)
	# Building shared library from static (static was compiled with -fPIC)
	$(CXX) -shared -o $@ -Wl,-soname=$@,--whole-archive $(LIBNAME).a -Wl,--no-whole-archive
	cp $@ $(KALDILIBDIR)
endif


$(BINFILES): $(LIBFILE) $(XDEPENDS)


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

%.so:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a *.so $(TESTFILES) $(BINFILES) tmp* *.tmp

$(TESTFILES): $(LIBFILE) $(XDEPENDS)

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(BINFILES) $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

