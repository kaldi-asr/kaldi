
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = sgmm2-init sgmm2-gselect sgmm2-acc-stats sgmm2-est sgmm2-sum-accs \
         sgmm2-align-compiled sgmm2-est-spkvecs sgmm2-post-to-gpost \
         sgmm2-acc-stats-gpost sgmm2-latgen-faster sgmm2-est-spkvecs-gpost \
         sgmm2-rescore-lattice sgmm2-copy sgmm2-info sgmm2-est-ebw \
         sgmm2-acc-stats2 sgmm2-comp-prexform sgmm2-est-fmllr sgmm2-project \
         sgmm2-latgen-faster-parallel

OBJFILES =



TESTFILES =


ADDLIBS = ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feat.a \
					../sgmm2/kaldi-sgmm2.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
					../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
					../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 

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
ifeq ($(KALDI_FLAVOR), dynamic)
	# $(CXX) -shared -o $(LIBFILE) -Wl,-soname=$(LIBFILE) $(OBJFILES)
	$(AR) -cru $(LIBNAME)-dyn.a $(OBJFILES)
	$(RANLIB) $(LIBNAME)-dyn.a
	$(CXX) -shared -o $@ -Wl,-soname=$@,--whole-archive $(LIBNAME)-dyn.a -Wl,--no-whole-archive
	cp $@ $(KALDILIBDIR)
else
	$(AR) -cru $(LIBNAME).a $(OBJFILES)
	$(RANLIB) $(LIBNAME).a
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

