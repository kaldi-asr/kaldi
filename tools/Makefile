# SHELL += -x

CXX = g++
CC = gcc         # used for sph2pipe
# CXX = clang++  # Uncomment these lines
# CC = clang     # to build with Clang.

# Note: OpenFst requires a relatively recent C++ compiler with C++11 support,
# e.g. g++ >= 4.7, Apple clang >= 5.0 or LLVM clang >= 3.3.
OPENFST_VERSION ?= 1.6.7
CUB_VERSION ?= 1.8.0

# Default features configured for OpenFST; can be overridden in the make command line.
OPENFST_CONFIGURE ?= --enable-static --enable-shared --enable-far --enable-ngram-fsts

OPENFST_VER_NUM := $(shell echo $(OPENFST_VERSION) | sed 's/\./ /g' | xargs printf "%d%02d%02d")
ifeq ("$(shell expr $(OPENFST_VER_NUM) \< 10600)","1")
    $(error OpenFst-$(OPENFST_VERSION) is not supported. \
            Supported versions: >= 1.6.0)
endif

all: check_required_programs sph2pipe sclite openfst cub
	@echo -e "\n\n"
	@echo "Warning: IRSTLM is not installed by default anymore. If you need IRSTLM"
	@echo "Warning: use the script extras/install_irstlm.sh"
	@echo "All done OK."

# make sure check_required_programs runs before anything else:
sph2pipe sclite openfst sctk: | check_required_programs

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
	rm -rf sph2pipe_v2.5/
	rm -rf sph2pipe_v2.5.tar.gz
	rm -rf sctk-2.4.10-20151007-1312Z.tar.bz2
	rm -rf openfst-$(OPENFST_VERSION).tar.gz
	rm -f openfst
	rm -rf libsndfile-1.0.25{,.tar.gz} BeamformIt-3.51{,.tgz}
	rm -f cub-$(CUB_VERSION).zip
	rm -rf cub-$(CUB_VERSION)
	rm -f cub


.PHONY: openfst # so target will be made even though "openfst" exists.
openfst: openfst_compiled openfst-$(OPENFST_VERSION)/lib
	-rm -f openfst
	-ln -s openfst-$(OPENFST_VERSION) openfst

.PHONY: openfst_compiled
openfst_compiled: openfst-$(OPENFST_VERSION)/Makefile
	$(MAKE) -C openfst-$(OPENFST_VERSION) install MAKEOVERRIDES=

openfst-$(OPENFST_VERSION)/lib: | openfst-$(OPENFST_VERSION)/Makefile
	-cd openfst-$(OPENFST_VERSION) && [ -d lib64 ] && [ ! -d lib ] && ln -s lib64 lib

# Add the -O flag to CXXFLAGS on cygwin as it can fix the compilation error
# "file too big".
ifeq ($(OSTYPE),cygwin)
  # Note: OSTYPE path is probably dead for latest cygwin64 (installed on 2016/11/11).
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else ifeq ($(OS),Windows_NT)
  # This new OS path is confirmed working on Windows 10 / Cygwin64.
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else
  openfst_add_CXXFLAGS =
endif

openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION) | check_required_programs
	cd openfst-$(OPENFST_VERSION)/ && \
	./configure --prefix=`pwd` $(OPENFST_CONFIGURE) CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) $(openfst_add_CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"


openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	wget -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
	wget -T 10 -t 3 http://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz

sclite: sclite_compiled

.PHONY: sclite_compiled
sclite_compiled: sctk sctk_configured
	cd sctk; \
	$(MAKE) CC="$(CC)" CXX="$(CXX)" all && $(MAKE) install && $(MAKE) doc

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

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
	cd sph2pipe_v2.5/; \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5: sph2pipe_v2.5.tar.gz
	tar --no-same-owner -xzf sph2pipe_v2.5.tar.gz

sph2pipe_v2.5.tar.gz:
	wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || \
	wget --no-check-certificate -T 10  https://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz

openblas: openblas_compiled

.PHONY: openblas_compiled

fortran_opt = $(shell gcc -v 2>&1 | perl -e '$$x = join(" ", <STDIN>); if($$x =~ m/target=\S+64\S+/) { print "BINARY=64"; }')


# note: you can uncomment the line that has USE_THREAD=1 and comment the line
# that has USE_THREAD=0 if you want Open Blas to use multiple threads.  then
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


.PHONY: cub
cub:
	wget -T 10 -t 3 -O cub-$(CUB_VERSION).zip https://github.com/NVlabs/cub/archive/$(CUB_VERSION).zip
	unzip -oq cub-$(CUB_VERSION).zip
	rm -f cub
	ln -s cub-$(CUB_VERSION) cub
