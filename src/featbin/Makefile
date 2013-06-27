
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-mfcc-feats compute-plp-feats compute-fbank-feats \
    compute-cmvn-stats add-deltas remove-mean apply-cmvn transform-feats \
    copy-feats compose-transforms splice-feats extract-segments subset-feats \
    feat-to-len feat-to-dim fmpe-apply-transform fmpe-acc-stats fmpe-init \
    fmpe-est fmpe-copy fmpe-sum-accs append-feats extend-transform-dim \
    get-full-lda-mat compute-spectrogram-feats extract-feature-segments \
    reverse-feats paste-feats select-feats subsample-feats process-pitch-feats \
    interpolate-pitch copy-feats-to-htk extract-rows

OBJFILES = 

all:  $(BINFILES)


TESTFILES =

ADDLIBS = ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
         ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
         ../base/kaldi-base.a

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
	gcc -shared -o $(LIBFILE) -Wl,-soname=$(LIBFILE) $(OBJFILES)
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

