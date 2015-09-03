all:

EXTRA_CXXFLAGS += -Wno-sign-compare -I ../../tools/portaudio/install/include
EXTRA_LDLIBS =

include ../kaldi.mk

UNAME=$(shell uname)
ifeq ($(UNAME), Darwin)
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
    EXTRA_LDLIBS += -framework CoreAudio -framework AudioToolbox -framework AudioUnit -framework CoreServices
endif

ifeq ($(UNAME), Linux)
  ifneq "$(wildcard ../../tools/portaudio/install/lib/libportaudio.a)" ""
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
  else 
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib64/libportaudio.a
  endif
  ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound -lrt
  else
    EXTRA_LDLIBS += -lrt
  endif
endif


TESTFILES = online-feat-test

OBJFILES = online-audio-source.o online-feat-input.o online-decodable.o online-faster-decoder.o onlinebin-util.o online-tcp-source.o

LIBNAME = kaldi-online

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../feat/kaldi-feat.a \
	../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a \
	../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a \
	../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

