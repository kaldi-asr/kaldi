all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../kaldi.mk

TESTFILES = 

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   faster-decoder.o lattice-tracking-decoder.o

LIBNAME = kaldi-decoder

ADDLIBS = ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
     ../sgmm/kaldi-sgmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../util/kaldi-util.a \
     ../base/kaldi-base.a ../matrix/kaldi-matrix.a 

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

