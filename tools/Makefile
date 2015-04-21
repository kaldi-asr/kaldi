# SHELL += -x

CXX = g++
# CXX = clang++  # Uncomment this line to build with Clang.

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

all: check_required_programs sph2pipe atlas irstlm_tgt sclite openfst

check_required_programs:
	extras/check_dependencies.sh

clean: openfst_cleaned sclite_cleaned  irstlm_cleaned

openfst_cleaned:
	$(MAKE) -C openfst-$(OPENFST_VERSION) clean

irstlm_cleaned:
	$(MAKE) -C irstlm clean 

sclite_cleaned:
	$(MAKE) -C sctk clean

distclean:
	rm -rf openfst-$(OPENFST_VERSION)/
	rm -rf sctk-2.4.9/
	rm -rf sctk
	rm -rf ATLAS/
	rm -rf irstlm/
	rm -rf sph2pipe_v2.5/
	rm -rf sph2pipe_v2.5.tar.gz
	rm -rf atlas3.8.3.tar.gz
	rm -rf sctk-2.4.9-20141015-1634Z.tar.bz2
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

# Disable shared-library creation on cygwin as it leads to compilation error
# "file too big".
openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION)/.patched
ifeq ($(OSTYPE),cygwin)
	cd openfst-$(OPENFST_VERSION)/; ./configure --prefix=`pwd` --enable-static --disable-shared --enable-far --enable-ngram-fsts CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"
else
	cd openfst-$(OPENFST_VERSION)/; ./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"
endif

# patches for openfst. openfst_gcc41up.patch is a patch for openfst to \
# support multi-threads when compile with g++ (gcc) version above 4.1
openfst-$(OPENFST_VERSION)/.patched: | openfst-$(OPENFST_VERSION)
	-cd openfst-$(OPENFST_VERSION)/src/include/fst; \
	patch -p0 -N < ../../../../extras/openfst-$(OPENFST_VERSION).patch;
	$(CXX) -dumpversion | awk '{if(NR==1 && $$1>"4.1") print "cd openfst-$(OPENFST_VERSION)/src/include/fst; patch -p0 -N < ../../../../extras/openfst_gcc41up.patch"}' | sh -
	touch $@

openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz: 
	wget http://openfst.cs.nyu.edu/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
	wget -T 10 -t 3 http://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz

sclite: sclite_compiled

.PHONY: sclite_compiled
sclite_compiled: sctk sctk_configured
	cd sctk; \
	$(MAKE) all && $(MAKE) install && $(MAKE) doc

sctk_configured: sctk sctk/.configured

sctk/.configured: sctk
	cd sctk; $(MAKE) config
	touch sctk/.configured

.PHONY: sctk
sctk: sctk-2.4.9-20141015-1634Z.tar.bz2
	tar xojf sctk-2.4.9-20141015-1634Z.tar.bz2
	rm -rf sctk && ln -s sctk-2.4.9 sctk

sctk-2.4.9-20141015-1634Z.tar.bz2:
	wget -T 10 -t 3 ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.9-20141015-1634Z.tar.bz2|| \
    wget --no-check-certificate -T 10 http://www.openslr.org/resources/4/sctk-2.4.9-20141015-1634Z.tar.bz2

irstlm_tgt: irstlm_compiled

.PHONY: irstlm_compiled
irstlm_compiled: irstlm/Makefile
	cd irstlm/; \
	make; $(MAKE) install	

irstlm/Makefile: irstlm
	cd irstlm; \
	    automake --version | grep 1.13.1 >/dev/null && \
         sed s:AM_CONFIG_HEADER:AC_CONFIG_HEADERS: <configure.in >configure.ac; \
	(./regenerate-makefiles.sh || ./regenerate-makefiles.sh) && \
	./configure --prefix `pwd` 

irstlm:
	svn -r 618 co --non-interactive --trust-server-cert https://svn.code.sf.net/p/irstlm/code/trunk irstlm

atlas: ATLAS/include/cblas.h

ATLAS/include/cblas.h: | atlas3.8.3.tar.gz
	tar xozf atlas3.8.3.tar.gz ATLAS/include

atlas3.8.3.tar.gz:
	wget -T 10  http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz || \
	wget --no-check-certificate -T 10 -t 3 http://www.danielpovey.com/files/kaldi/atlas3.8.3.tar.gz

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
	cd sph2pipe_v2.5/; \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5: sph2pipe_v2.5.tar.gz
	tar xzf sph2pipe_v2.5.tar.gz

sph2pipe_v2.5.tar.gz:
	wget --no-check-certificate -T 10  https://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz || \
	wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz

openblas: openblas_compiled

.PHONY: openblas_compiled

fortran_opt = $(shell gcc -v 2>&1 | perl -e '$$x = join(" ", <STDIN>); if($$x =~ m/target=\S+64\S+/) { print "BINARY=64"; }')


# note: you can change the USE_THREAD=0 option below to USE_THREAD=1
# if you want Open Blas to use multiple threads.  then you could set,
# for example, OPENBLAS_NUM_THREADS=2 in your path.sh so that the
# runtime knows how many threads to use.
# Note: if you ever get the error "Program is Terminated. Because you tried to
# allocate too many memory regions.", this is because OpenBLAS has a fixed
# buffer size for the number of threads and you might have gone beyond that.  It
# may possibly help to add e.g. NUM_THREADS=64 to the command line below (after
# $(MAKE)).
openblas_compiled:
	-git clone git://github.com/xianyi/OpenBLAS
	cd OpenBLAS; sed 's:# FCOMMON_OPT = -frecursive:FCOMMON_OPT = -frecursive:' < Makefile.rule >tmp && mv tmp Makefile.rule
	$(MAKE) PREFIX=`pwd`/OpenBLAS/install FC=gfortran $(fortran_opt) DEBUG=1 USE_THREAD=1 -C OpenBLAS all install

# libsndfile needed by beamformit,
libsndfile: libsndfile-1.0.25.tar.gz libsndfile-1.0.25/lib/libsndfile.so
libsndfile-1.0.25.tar.gz:
	wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
libsndfile-1.0.25/lib/libsndfile.so:
	tar xvf libsndfile-1.0.25.tar.gz && cd libsndfile-1.0.25/ && \
	./configure --prefix=$(PWD)/libsndfile-1.0.25/ && make && make install

# beamforming tool,
beamformit: libsndfile beamformit-3.51

.PHONY: beamformit-3.51

beamformit-3.51: beamformit-3.51.tgz
	tar -xozf BeamformIt-3.51.tgz; \
	cd BeamformIt-3.51; patch -p1 -N < ../extras/beamformit.patch; cmake . ; make

beamformit-3.51.tgz:
	wget -c -T 10 http://www.xavieranguera.com/beamformit/releases/BeamformIt-3.51.tgz

