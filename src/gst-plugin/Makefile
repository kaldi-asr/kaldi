
all:

EXTRA_CXXFLAGS = -fPIC -Wno-sign-compare -I /usr/include/gstreamer-1.0 \
	-I /usr/include/glib-2.0 -I /usr/lib/x86_64-linux-gnu/glib-2.0/include \
	-I ../../tools/portaudio/install/include
EXTRA_LDLIBS =

include ../kaldi.mk

OBJFILES = gst-audio-source.o gst-online-gmm-decode-faster.o

LIBNAME=gstkaldi

LIBFILE = lib$(LIBNAME).so
BINFILES= $(LIBFILE)

all: $(LIBFILE)

GST_LIBS = -pthread -lgstbase-1.0 -lgstcontroller-1.0 -lgstreamer-1.0 -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lrt -lglib-2.0
#GST_PLUGIN_LDFLAGS = -module -avoid-version -export-symbols-regex _*\(gst_\|Gst\|GST_\).*

ADDLIBS=../online/kaldi-online.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a \
 ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a \
 ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a \
 ../../tools/portaudio/install/lib/libportaudio.a

GSTKALDI_SO_LDFLAGS = -lasound $(LDFLAGS) $(LDLIBS) 

# MKL libs required for shared library
ifdef MKLROOT
GSTKALDI_SO_LDFLAGS+=-lmkl_p4n -lmkl_def
endif

$(LIBFILE): $(OBJFILES)
	$(CXX) -shared -DPIC -o $(LIBFILE) -Wl,-soname=$(LIBFILE) $(GST_LIBS) $(GSTKALDI_SO_LDFLAGS)  \
	-Wl,-whole-archive $(ADDLIBS) -Wl,-no-whole-archive $(OBJFILES)

 
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
