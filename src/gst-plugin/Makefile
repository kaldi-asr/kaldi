all:

include ../kaldi.mk
ifneq ($(KALDI_FLAVOR), dynamic)
$(error Kaldi must compiled with dynamic libraries support. Run configure with --shared flag. )
endif

EXTRA_CXXFLAGS += -Wno-sign-compare -I ../../tools/portaudio/install/include 
EXTRA_CXXFLAGS += $(shell pkg-config --cflags gstreamer-1.0)
EXTRA_CXXFLAGS += $(shell pkg-config --cflags glib-2.0)

EXTRA_LDLIBS += -lgstbase-1.0 -lgstcontroller-1.0 -lgmodule-2.0 -lgthread-2.0
EXTRA_LDLIBS += $(shell pkg-config --libs gstreamer-1.0)
EXTRA_LDLIBS += $(shell pkg-config --libs glib-2.0)


#Kaldi shared libraries required by the GStreamer plugin
EXTRA_LDLIBS += -lkaldi-online -lkaldi-lat -lkaldi-decoder -lkaldi-feat -lkaldi-transform \
 -lkaldi-gmm -lkaldi-hmm \
 -lkaldi-tree -lkaldi-matrix  -lkaldi-util -lkaldi-base 


OBJFILES = gst-audio-source.o gst-online-gmm-decode-faster.o

LIBNAME=gstonlinegmmdecodefaster

LIBFILE = lib$(LIBNAME).so
BINFILES= $(LIBFILE)

all: $(LIBFILE)

EXTRA_LDLIBS += ../../tools/portaudio/install/lib/libportaudio.a
ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
endif

# Library so name and rpath
CXX_VERSION=$(shell $(CXX) --version 2>/dev/null)
ifneq (,$(findstring clang, $(CXX_VERSION)))
    # clang++ linker
    EXTRA_LDLIBS +=  -Wl,-install_name,$(LIBFILE) -Wl,-rpath,$(KALDILIBDIR)
else
    # g++ linker
    EXTRA_LDLIBS +=  -Wl,-soname=$(LIBFILE) -Wl,--no-as-needed -Wl,-rpath=$(KALDILIBDIR) -lrt -pthread
endif

$(LIBFILE): $(OBJFILES)
	$(CXX) -shared -DPIC -o $(LIBFILE) -L$(KALDILIBDIR) $(EXTRA_LDLIBS) $(LDLIBS) $(LDFLAGS) \
	  $(OBJFILES)
 
kaldimarshal.h: kaldimarshal.list
	glib-genmarshal --header --prefix=kaldi_marshal kaldimarshal.list > kaldimarshal.h.tmp
	mv kaldimarshal.h.tmp kaldimarshal.h

kaldimarshal.cc: kaldimarshal.list
	echo "#include \"glib-object.h\"" >> kaldimarshal.c.tmp
	echo "#include \"kaldimarshal.h\"" >> kaldimarshal.c.tmp
	glib-genmarshal --body --prefix=kaldi_marshal kaldimarshal.list >> kaldimarshal.c.tmp
	mv kaldimarshal.c.tmp kaldimarshal.cc
 
clean: 
	-rm -f *.o *.a $(TESTFILES) $(BINFILES) kaldimarshal.h kaldimarshal.cc
 
depend: kaldimarshal.h kaldimarshal.cc
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

-include .depend.mk


