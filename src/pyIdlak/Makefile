
include ../kaldi.mk

OBJFILES = python-api.o pyIdlak_wrap.o

LIBNAME = _pyIdlak

ADDLIBS = ../idlaktxp/idlak-txp.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

EXTRA_CXXFLAGS = -fPIC -I$(PYTHONDEVINC) -I$(PCREROOT)/include -I$(EXPATROOT)/include -I$(PUJIXMLROOT)/src
EXTRA_LDLIBS = -lpython2.7 -Wl,-rpath,"$(PCREROOT)/lib" $(PCREROOT)/lib/libpcre.so $(EXPATROOT)/lib/libexpat.so

include ../makefiles/default_rules.mk

ifeq ($(KALDI_FLAVOR), dynamic)
  STATICLIB =
else
  STATICLIB = _pyIdlak.a
endif

.PHONY: wrapper
wrapper:
	$(SWIG) -c++ -python -o pyIdlak_wrap.cc pyIdlak.i
pyIdlak_wrap.o: wrapper

sharedlib: $(STATICLIB)
ifeq ($(KALDI_FLAVOR), dynamic)
  ifeq ($(shell uname), Darwin)
	ln -fs $(KALDILIBDIR)/lib_pyIdlak.dylib _pyIdlak.dylib
  else ifeq ($(shell uname), Linux)
	ln -fs $(KALDILIBDIR)/lib_pyIdlak.so _pyIdlak.so
  else  # Platform not supported
	$(error Dynamic libraries not supported on this platform '$(shell uname)'. Python wrapper cannot be built.)
  endif
else
  ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o _pyIdlak.dylib -install_name @rpath/_pyIdlak.a $(LDFLAGS) $(LDLIBS)
  else ifeq ($(shell uname), Linux)
	# Building shared library from static (static was compiled with -fPIC)
	$(CXX) -shared -o _pyIdlak.so -Wl,--no-undefined -Wl,--as-needed \
		-Wl,-soname=_pyIdlak.so,--whole-archive _pyIdlak.a $(ADDLIBS) $(PUJIXMLROOT)/scripts/libpugixml.a -Wl,--no-whole-archive $(LDFLAGS) $(LDLIBS)
  else  # Platform not supported
	$(error Dynamic libraries not supported on this platform '$(shell uname)'. Python wrapper cannot be built.)
  endif
endif

clean:
	-rm -f *.o *.a *.so pyIdlak.py pyIdlak.pyc pyIdlak_wrap.cc
	@echo "REQUIRED FOR MAKE DO NOT DELETE!" > pyIdlak_wrap.cc

all: sharedlib
