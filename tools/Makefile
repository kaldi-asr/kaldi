# SHELL += -x

CXX ?= g++
CC ?= gcc        # used for sph2pipe
# CXX = clang++  # Uncomment these lines...
# CC = clang     # ...to build with Clang.

WGET ?= wget

# Note: OpenFst requires a relatively recent C++ compiler with C++11 support,
# e.g. g++ >= 4.7, Apple clang >= 5.0 or LLVM clang >= 3.3.
OPENFST_VERSION ?= 1.6.7
CUB_VERSION ?= 1.8.0
OPENBLAS_VERSION ?= 0.3.5
SCTK_VERSION_PARTIAL ?= 2.4.10
SCTK_VERSION ?= $(SCTK_VERSION_PARTIAL)-20151007-1312Z
SPH2PIPE_VERSION = v2.5

# Default features configured for OpenFST; can be overridden in the make command line.
OPENFST_CONFIGURE ?= --enable-static --enable-shared --enable-far --enable-ngram-fsts --enable-lookahead-fsts --with-pic

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
	rm -rf sctk-$(SCTK_VERSION_PARTIAL)/
	rm -rf sctk
	rm -rf sph2pipe_$(SPH2PIPE_VERSION)/
	rm -rf sph2pipe_$(SPH2PIPE_VERSION).tar.gz
	rm -rf sctk-$(SCTK_VERSION).tar.bz2
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
  openfst_add_CXXFLAGS = -g -O2 -Wa,-mbig-obj
else ifeq ($(OS),Windows_NT)
  # This new OS path is confirmed working on Windows 10 / Cygwin64.
  openfst_add_CXXFLAGS = -g -O2 -Wa,-mbig-obj
else
  openfst_add_CXXFLAGS = -g -O2
endif

openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION) | check_required_programs
	cd openfst-$(OPENFST_VERSION)/ && \
	./configure --prefix=`pwd` $(OPENFST_CONFIGURE) CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) $(openfst_add_CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"


openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
		cp -p "$(DOWNLOAD_DIR)/openfst-$(OPENFST_VERSION).tar.gz" .; \
	else \
		$(WGET) -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
		$(WGET) -T 10 -t 3 -c https://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz; \
	fi

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
sctk: sctk-$(SCTK_VERSION).tar.bz2
	tar xojf sctk-$(SCTK_VERSION).tar.bz2 || \
	tar --exclude '*NONE*html' -xvojf sctk-$(SCTK_VERSION).tar.bz2
	rm -rf sctk && ln -s sctk-$(SCTK_VERSION_PARTIAL) sctk

sctk-$(SCTK_VERSION).tar.bz2:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
		cp -p "$(DOWNLOAD_DIR)/sctk-$(SCTK_VERSION).tar.bz2" .; \
	else \
		$(WGET) -T 10 https://www.openslr.org/resources/4/sctk-$(SCTK_VERSION).tar.bz2; \
	fi

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_$(SPH2PIPE_VERSION)/sph2pipe

sph2pipe_$(SPH2PIPE_VERSION)/sph2pipe: | sph2pipe_$(SPH2PIPE_VERSION)
	cd sph2pipe_$(SPH2PIPE_VERSION)/ && \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_$(SPH2PIPE_VERSION): sph2pipe_$(SPH2PIPE_VERSION).tar.gz
	tar --no-same-owner -xzf sph2pipe_$(SPH2PIPE_VERSION).tar.gz

sph2pipe_$(SPH2PIPE_VERSION).tar.gz:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
		cp -p "$(DOWNLOAD_DIR)/sph2pipe_$(SPH2PIPE_VERSION).tar.gz" .; \
	else \
		$(WGET) -T 10 -t 3 https://www.openslr.org/resources/3/sph2pipe_$(SPH2PIPE_VERSION).tar.gz || \
		$(WGET) -T 10 -c https://sourceforge.net/projects/kaldi/files/sph2pipe_$(SPH2PIPE_VERSION).tar.gz; \
	fi

.PHONY: cub
cub:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
		cp -p "$(DOWNLOAD_DIR)/cub-$(CUB_VERSION).zip" .; \
	else \
		$(WGET) -T 10 -t 3 -O cub-$(CUB_VERSION).zip https://github.com/NVlabs/cub/archive/$(CUB_VERSION).zip; \
	fi
	unzip -oq cub-$(CUB_VERSION).zip
	rm -f cub
	ln -s cub-$(CUB_VERSION) cub

# OpenBLAS is not compiled by default; you can run `make openblas` to build it,
# but you should probably just call extras/install_openblas.sh; this is
# here for legacy reasons.
.PHONY: openblas
openblas:
	extras/install_openblas.sh
