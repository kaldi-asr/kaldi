# SHELL += -x

CXX = g++
# CXX = clang++  # Uncomment this line to build with Clang.
CC = gcc    # used for sph2pipe


OPENFST_VERSION = 1.3.4
# Uncomment the next line to build with OpenFst-1.4.1.
# OPENFST_VERSION = 1.4.1
# Note: OpenFst >= 1.4 requires C++11 support, hence you will need to use a
# relatively recent C++ compiler, e.g. gcc >= 4.6, clang >= 3.0.

# On Mac OS 10.9+, clang defaults to the new c++ standard library libc++.
# Since OpenFst-1.3 uses stuff from the tr1 namespace, we need to tell clang
# to use libstdc++ instead.
ifeq ($(OPENFST_VERSION), 1.3.4)
  COMPILER = $(shell $(CXX) -v 2>&1 )
  ifeq ($(findstring clang,$(COMPILER)),clang)
    CXXFLAGS += -stdlib=libstdc++
    LDFLAGS += -stdlib=libstdc++
  endif
else
  ifneq ($(OPENFST_VERSION), 1.4.1)
    $(error OpenFst version $(OPENFST_VERSION) is not supported. \
            Supported versions: 1.3.4, 1.4.1)
  endif
endif

all: check_required_programs sph2pipe atlas sclite openfst
	@echo -e "\n\n"
	@echo "Warning: IRSTLM is not installed by default anymore. If you need IRSTLM"
	@echo "Warning: use the script extras/install_irstlm.sh"
	@echo "All done OK."

# make sure check_required_programs runs before anything else:
sph2pipe atlas sclite openfst sctk: | check_required_programs

check_required_programs:
	extras/check_dependencies.sh

clean: openfst_cleaned sclite_cleaned

openfst_cleaned:
	$(MAKE) -C openfst-$(OPENFST_VERSION) clean


sclite_cleaned:
	$(MAKE) -C sctk clean

distclean:
	rm -rf openfst-$(OPENFST_VERSION)/
	rm -rf sctk-2.4.10/
	rm -rf sctk
	rm -rf ATLAS/
	rm -rf sph2pipe_v2.5/
	rm -rf sph2pipe_v2.5.tar.gz
	rm -rf atlas3.8.3.tar.gz
	rm -rf sctk-2.4.10-20151007-1312Z.tar.bz2
	rm -rf openfst-$(OPENFST_VERSION).tar.gz
	rm -f openfst
	rm -rf libsndfile-1.0.25{,.tar.gz} BeamformIt-3.51{,.tgz}


.PHONY: openfst # so target will be made even though "openfst" exists.
openfst: openfst_compiled openfst-$(OPENFST_VERSION)/lib
	-rm -f openfst
	-ln -s openfst-$(OPENFST_VERSION) openfst

.PHONY: openfst_compiled
openfst_compiled: openfst-$(OPENFST_VERSION)/Makefile
	cd openfst-$(OPENFST_VERSION)/ && \
	$(MAKE) install

openfst-$(OPENFST_VERSION)/lib: | openfst-$(OPENFST_VERSION)/Makefile
	-cd openfst-$(OPENFST_VERSION) && [ -d lib64 ] && [ ! -d lib ] && ln -s lib64 lib

# Add the -O flag to CXXFLAGS on cygwin as it can fix the compilation error
# "file too big".
openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION)/.patched | check_required_programs
# Note: OSTYPE path is probably dead for latest cygwin64 (installed on 2016/11/11).
ifeq ($(OSTYPE),cygwin)
	cd openfst-$(OPENFST_VERSION)/; ./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts CXX=$(CXX) CXXFLAGS="$(CXXFLAGS) -O -Wa,-mbig-obj" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"
# This new OS path is confirmed working on Windows 10 / Cygwin64
else ifeq ($(OS),Windows_NT)
	cd openfst-$(OPENFST_VERSION)/; ./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts CXX=$(CXX) CXXFLAGS="$(CXXFLAGS) -O -Wa,-mbig-obj" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"
else
	# ppc64le needs the newsted config.guess to be correctly indentified
	[ "$(shell uname -p)" = "ppc64le" ] && wget -O openfst-$(OPENFST_VERSION)/config.guess \
		"http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD" || \
		echo "config.guess unchanged"
	cd openfst-$(OPENFST_VERSION)/; ./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"
endif

# patches for openfst. openfst_gcc41up.patch is a patch for openfst to \
# support multi-threads when compile with g++ (gcc) version above 4.1
openfst-$(OPENFST_VERSION)/.patched: | openfst-$(OPENFST_VERSION)
	cd openfst-$(OPENFST_VERSION)/; \
	patch -p1 -N < ../extras/openfst-$(OPENFST_VERSION).patch;
	$(CXX) -dumpversion | awk '{if(NR==1 && $$1>"4.1") print "cd openfst-$(OPENFST_VERSION)/src/include/fst; patch -c -p0 -N < ../../../../extras/openfst_gcc41up.patch"}' | sh -
	touch $@

openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	wget --tries=1 -T 5 http://openfst.cs.nyu.edu/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
	wget -T 10 -t 3 http://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz

sclite: sclite_compiled

.PHONY: sclite_compiled
sclite_compiled: sctk sctk_configured
	cd sctk; \
	$(MAKE) CC=$(CC) CXX=$(CXX) all && $(MAKE) install && $(MAKE) doc

sctk_configured: sctk sctk/.configured

sctk/.configured: sctk
	cd sctk; $(MAKE) config
	touch sctk/.configured

.PHONY: sctk
sctk: sctk-2.4.10-20151007-1312Z.tar.bz2
	tar xojf sctk-2.4.10-20151007-1312Z.tar.bz2 || \
      tar --exclude '*NONE*html' -xvojf sctk-2.4.10-20151007-1312Z.tar.bz2
	rm -rf sctk && ln -s sctk-2.4.10 sctk

sctk-2.4.10-20151007-1312Z.tar.bz2:
	wget -T 10 -t 3 ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.10-20151007-1312Z.tar.bz2|| \
	wget --no-check-certificate -T 10 http://www.openslr.org/resources/4/sctk-2.4.10-20151007-1312Z.tar.bz2


atlas: ATLAS/include/cblas.h

ATLAS/include/cblas.h: | atlas3.8.3.tar.gz
	tar xozf atlas3.8.3.tar.gz ATLAS/include

atlas3.8.3.tar.gz:
	wget -T 10 ftp://ftp.vim.org/vol/2/metalab/distributions/tinycorelinux/2.x/tce/src/libatlas/atlas3.8.3.tar.gz || \
    wget -T 10 http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz || \
	wget --no-check-certificate -T 10 -t 3 http://www.danielpovey.com/files/kaldi/atlas3.8.3.tar.gz

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
	cd sph2pipe_v2.5/; \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5: sph2pipe_v2.5.tar.gz
	tar xzf sph2pipe_v2.5.tar.gz

sph2pipe_v2.5.tar.gz:
	wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || \
	wget --no-check-certificate -T 10  https://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz

openblas: openblas_compiled

.PHONY: openblas_compiled

fortran_opt = $(shell gcc -v 2>&1 | perl -e '$$x = join(" ", <STDIN>); if($$x =~ m/target=\S+64\S+/) { print "BINARY=64"; }')


# note: you can uncomment the line that has USE_THREAD=1 and comment the line
# that has USE_THREADE=0 if you want Open Blas to use multiple threads.  then
# you could set, for example, OPENBLAS_NUM_THREADS=2 in your path.sh so that the
# runtime knows how many threads to use.  Note: if you ever get the error
# "Program is Terminated. Because you tried to allocate too many memory
# regions.", this is because OpenBLAS has a fixed buffer size controlled by the
# Makefile option NUM_THREADS; I believe this limits the product of number of
# program threads that are calling BLAS by the shell variable
# OPENBLAS_NUM_THREADS.  In that case it might help to increase the NUM_THREADS
# option.
openblas_compiled:
	echo "Note: see tools/Makefile for options regarding OpenBLAS compilation"
	-git clone https://github.com/xianyi/OpenBLAS.git
	-cd OpenBLAS; git pull
	cd OpenBLAS; sed 's:# FCOMMON_OPT = -frecursive:FCOMMON_OPT = -frecursive:' < Makefile.rule >tmp && mv tmp Makefile.rule
	# $(MAKE) PREFIX=`pwd`/OpenBLAS/install FC=gfortran $(fortran_opt) DEBUG=1 USE_THREAD=1 NUM_THREADS=64 -C OpenBLAS all install
	$(MAKE) PREFIX=`pwd`/OpenBLAS/install FC=gfortran $(fortran_opt) DEBUG=1 USE_THREAD=0 -C OpenBLAS all install
