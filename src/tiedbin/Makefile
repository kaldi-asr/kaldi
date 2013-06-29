
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = \
  tied-diag-gmm-init-mono \
  tied-diag-gmm-acc-stats-ali \
  tied-diag-gmm-est \
  tied-diag-gmm-align-compiled \
  tied-diag-gmm-decode-simple \
  tied-diag-gmm-decode-faster \
  tied-diag-gmm-init-model \
  smooth-stats-diag \
  tied-full-gmm-init-mono \
  tied-full-gmm-acc-stats-ali \
  tied-full-gmm-est \
  tied-full-gmm-align-compiled \
  tied-full-gmm-decode-simple \
  tied-full-gmm-decode-faster \
  tied-full-gmm-init-model \
  smooth-stats-full \
  init-tied-codebooks \
  tied-lbg \
  full-to-diag

OBJFILES =



TESTFILES =


ADDLIBS = ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../tied/kaldi-tied.a \
					../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
					../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

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

