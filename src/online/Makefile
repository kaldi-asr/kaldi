all:

EXTRA_CXXFLAGS += -Wno-sign-compare -I ../../tools/portaudio/install/include
EXTRA_LDLIBS =

include ../kaldi.mk

TESTFILES = online-feat-test

OBJFILES = online-audio-source.o online-feat-input.o online-decodable.o online-faster-decoder.o onlinebin-util.o online-tcp-source.o

LIBNAME = kaldi-online

include ../makefiles/default_rules.mk
