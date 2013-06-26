
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = init-ubm sgmm-align-compiled sgmm-acc-stats-ali \
           sgmm-sum-accs sgmm-est sgmm-decode-faster sgmm-init sgmm-gselect \
           sgmm-est-fmllr sgmm-acc-stats sgmm-est-spkvecs sgmm-post-to-gpost \
           sgmm-acc-stats-gpost sgmm-est-spkvecs-gpost sgmm-comp-prexform \
           sgmm-est-fmllr-gpost sgmm-acc-fmllrbasis-ali sgmm-est-fmllrbasis \
           sgmm-calc-distances sgmm-normalize sgmm-latgen-simple \
           sgmm-latgen-faster sgmm-rescore-lattice sgmm-copy sgmm-write-ubm \
           sgmm-mixup sgmm-info sgmm-acc-tree-stats sgmm-sum-tree-stats \
           sgmm-build-tree sgmm-cluster-phones sgmm-init-from-tree-stats \
           sgmm-est-ebw sgmm-acc-stats2 sgmm-est-multi

OBJFILES =



TESTFILES =


ADDLIBS = ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a \
          ../sgmm/kaldi-sgmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
					../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
					../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 

ifeq ($(KALDI_FLAVOR), dynamic)
ifdef LIBNAME
LIBFILE = lib$(LIBNAME).so
LDLIBS  += -l$(LIBNAME)
endif
LDFLAGS += $(foreach dep,$(ADDLIBS), -L$(dir $(dep)) )
LDFLAGS += -L.
LDFLAGS += -Wl,-rpath=$(shell readlink -f ../dynlib)
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
	gcc -shared -o $(LIBFILE) -Wl,-soname=$(LIBFILE) $(OBJFILES)
	cp $@ ../dynlib
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

