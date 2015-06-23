
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare

TESTFILES = kaldi-lattice-test push-lattice-test minimize-lattice-test \
      determinize-lattice-pruned-test 

OBJFILES = kaldi-lattice.o lattice-functions.o word-align-lattice.o \
	   phone-align-lattice.o word-align-lattice-lexicon.o sausages.o \
        push-lattice.o minimize-lattice.o determinize-lattice-pruned.o \
				confidence.o

LIBNAME = kaldi-lat

ADDLIBS = ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk

# Overriding the default library rule
# Added 2015-06-22 in connection with creating a standalone kws lib
# It's purpose is to make the transition more seamless for users
# Will be removed in a half a year or so.
$(LIBFILE): $(OBJFILES)
	$(AR) -d $(LIBNAME).a kws-functions.o
	$(AR) -cru $(LIBNAME).a $(OBJFILES)
	$(RANLIB) $(LIBNAME).a
ifeq ($(KALDI_FLAVOR), dynamic)
ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o $@ -install_name @rpath/$@ -framework Accelerate $(LDFLAGS) $(XLDLIBS) $(OBJFILES) $(LDLIBS)
	rm -f $(KALDILIBDIR)/$@; ln -s $(shell pwd)/$@ $(KALDILIBDIR)/$@
else
ifeq ($(shell uname), Linux)
	# Building shared library from static (static was compiled with -fPIC)
	$(CXX) -shared -o $@ -Wl,--no-undefined -Wl,--as-needed  -Wl,-soname=$@,--whole-archive $(LIBNAME).a -Wl,--no-whole-archive  $(LDFLAGS) $(XDEPENDS) $(LDLIBS)
	rm -f $(KALDILIBDIR)/$@; ln -s $(shell pwd)/$@ $(KALDILIBDIR)/$@
	#cp $@ $(KALDILIBDIR)
else  # Platform not supported
	$(error Dynamic libraries not supported on this platform. Run configure with --static flag. )
endif
endif
endif

