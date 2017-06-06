
SHELL := /bin/bash

ifeq ($(KALDI_FLAVOR), dynamic)
  ifeq ($(shell uname), Darwin)
    ifdef LIBNAME
      LIBFILE = lib$(LIBNAME).dylib
    endif
    LDFLAGS += -Wl,-rpath -Wl,$(KALDILIBDIR)
    EXTRA_LDLIBS += $(foreach dep,$(ADDLIBS), $(dir $(dep))lib$(notdir $(basename $(dep))).dylib)
  else ifeq ($(shell uname), Linux)
    ifdef LIBNAME
      LIBFILE = lib$(LIBNAME).so
    endif
    LDFLAGS += -Wl,-rpath=$(shell readlink -f $(KALDILIBDIR))
    EXTRA_LDLIBS += $(foreach dep,$(ADDLIBS), $(dir $(dep))lib$(notdir $(basename $(dep))).so)
  else  # Platform not supported
    $(error Dynamic libraries not supported on this platform. Run configure with --static flag.)
  endif
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
	$(CXX) -dynamiclib -o $@ -install_name @rpath/$@ $(LDFLAGS) $(OBJFILES) $(LDLIBS)
	rm -f $(KALDILIBDIR)/$@; ln -s $(shell pwd)/$@ $(KALDILIBDIR)/$@
else ifeq ($(shell uname), Linux)
	# Building shared library from static (static was compiled with -fPIC)
	$(CXX) -shared -o $@ -Wl,--no-undefined -Wl,--as-needed  -Wl,-soname=$@,--whole-archive $(LIBNAME).a -Wl,--no-whole-archive $(LDFLAGS) $(LDLIBS)
	rm -f $(KALDILIBDIR)/$@; ln -s $(shell pwd)/$@ $(KALDILIBDIR)/$@
else  # Platform not supported
	$(error Dynamic libraries not supported on this platform. Run configure with --static flag.)
endif
endif

# By default (GNU) make uses the C compiler $(CC) for linking object files even
# if they were compiled from a C++ source. Below redefinition forces make to
# use the C++ compiler $(CXX) instead.
LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)

ifeq ($(KALDI_FLAVOR), dynamic)
$(BINFILES): $(LIBFILE)
else
$(BINFILES): $(LIBFILE) $(XDEPENDS)
endif

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

%.so:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a *.so $(TESTFILES) $(BINFILES) $(TESTOUTPUTS) tmp* *.tmp *.testlog

distclean: clean
	-rm -f .depend.mk

ifeq ($(KALDI_FLAVOR), dynamic)
$(TESTFILES): $(LIBFILE)
else
$(TESTFILES): $(LIBFILE) $(XDEPENDS)
endif

test_compile: $(TESTFILES)

test: test_compile
	@{ result=0;			\
	for x in $(TESTFILES); do	\
	  printf "Running $$x ...";	\
      timestamp1=$$(date +"%s"); \
	  ./$$x >$$x.testlog 2>&1;	\
      ret=$$? \
      timestamp2=$$(date +"%s"); \
      time_taken=$$[timestamp2-timestamp1]; \
	  if [ $$ret -ne 0 ]; then \
	    echo " $${time_taken}s... FAIL $$x"; \
	    result=1;			\
	    if [ -n "$TRAVIS" ] && [ -f core ] && command -v gdb >/dev/null 2>&1; then	\
	      gdb $$x core -ex "thread apply all bt" -batch >>$$x.testlog 2>&1;		\
	      rm -rf core;		\
	    fi;				\
	  else				\
	    echo " $${time_taken}s... SUCCESS $$x";		\
	    rm -f $$x.testlog;		\
	  fi;				\
	done;				\
	exit $$result; }

# Rules that enable valgrind debugging ("make valgrind")

valgrind: .valgrind

.valgrind: $(TESTFILES)
	echo -n > valgrind.out
	for x in $(TESTFILES); do \
		echo $$x >>valgrind.out; \
		valgrind ./$$x >/dev/null 2>> valgrind.out; \
	done
	! ( grep 'ERROR SUMMARY' valgrind.out | grep -v '0 errors' )
	! ( grep 'definitely lost' valgrind.out | grep -v -w 0 )
	rm valgrind.out
	touch .valgrind

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend
-include .depend.mk
