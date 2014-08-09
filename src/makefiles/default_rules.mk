ifeq ($(KALDI_FLAVOR), dynamic)
  ifeq ($(shell uname), Darwin)
    XLDLIBS := $(LDLIBS)
    ifdef LIBNAME
      LIBFILE = lib$(LIBNAME).dylib
      #LDLIBS  += -l$(LIBNAME)
    endif
    LDFLAGS += -L$(KALDILIBDIR) -Wl,-rpath -Wl,$(KALDILIBDIR)
    XDEPENDS = $(foreach dep,$(ADDLIBS), $(dir $(dep))/lib$(notdir $(basename $(dep))).dylib )
    XLDLIBS += $(foreach dep,$(ADDLIBS), -l$(notdir $(basename $(dep))) )
  else
    ifeq ($(shell uname), Linux)
      ifdef LIBNAME
        LIBFILE = lib$(LIBNAME).so
        #LDLIBS  += -l$(LIBNAME)
      endif
      LDFLAGS += -Wl,-rpath=$(shell readlink -f $(KALDILIBDIR)) -L.
      LDFLAGS += $(foreach dep,$(ADDLIBS), -L$(dir $(dep)) )
      XDEPENDS = $(foreach dep,$(ADDLIBS), $(dir $(dep))/lib$(notdir $(basename $(dep))).so )
    else  # Platform not supported
      $(error Dynamic libraries not supported on this platform. Run configure with --static flag. )
    endif
  endif
  LDLIBS  += $(foreach dep,$(ADDLIBS), -l$(notdir $(basename $(dep))) )
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
	-rm -f *.o *.a *.so $(TESTFILES) $(BINFILES) $(TESTOUTPUTS) tmp* *.tmp

$(TESTFILES): $(LIBFILE) $(XDEPENDS)

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do printf "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL $$x"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(BINFILES) $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend
-include .depend.mk

