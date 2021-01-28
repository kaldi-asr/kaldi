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
# No '?=', since there exists only one version of sph2pipe.
SPH2PIPE_VERSION = 2.5
# SCTK official repo does not have version tags. Here's the mapping:
# 2.4.9 = 659bc36; 2.4.10 = d914e1b; 2.4.11 = 20159b5.
SCTK_GITHASH = 20159b5

# Default features configured for OpenFST; can be overridden in the make command line.
OPENFST_CONFIGURE ?= --enable-static --enable-shared --enable-far \
                     --enable-ngram-fsts --enable-lookahead-fsts --with-pic

OPENFST_VER_NUM := $(shell echo $(OPENFST_VERSION) | sed 's/\./ /g' | xargs printf "%d%02d%02d")
ifeq ("$(shell expr $(OPENFST_VER_NUM) \< 10600)","1")
    $(error OpenFst-$(OPENFST_VERSION) is not supported. \
            Supported versions: >= 1.6.0)
endif

.PHONY: all clean distclean
all: check_required_programs cub openfst sctk_made sph2pipe
	@echo -e "\n\n"
	@echo "Warning: IRSTLM is not installed by default anymore. If you need IRSTLM"
	@echo "Warning: use the script extras/install_irstlm.sh"
	@echo "All done OK."

# make sure check_required_programs runs before anything else:
all cub openfst sctk_made sph2pipe: | check_required_programs

check_required_programs:
	extras/check_dependencies.sh

clean: openfst_cleaned sctk_cleaned

distclean: clean
	rm -rf openfst openfst-$(OPENFST_VERSION) \
	       openfst-$(OPENFST_VERSION).tar.gz
	rm -rf sctk sctk-$(SCTK_GITHASH) \
	       sctk-$(SCTK_GITHASH).tar.gz
	rm -rf sph2pipe sph2pipe-$(SPH2PIPE_VERSION) \
	       sph2pipe-$(SPH2PIPE_VERSION).tar.gz
	rm -rf libsndfile-1.0.25 libsndfile-1.0.25.tar.gz \
	       BeamformIt-3.51 BeamformIt-3.51.tgz
	rm -rf cub cub-$(CUB_VERSION) \
	       cub-$(CUB_VERSION).zip cub-$(CUB_VERSION).tar.gz

#=== OpenFST ===================================================================

.PHONY: openfst openfst_compiled openfst_cleaned

openfst: openfst_compiled openfst-$(OPENFST_VERSION)/lib
	rm -f openfst
	ln -s openfst-$(OPENFST_VERSION) openfst

openfst_compiled: openfst-$(OPENFST_VERSION)/Makefile
	$(MAKE) -C openfst-$(OPENFST_VERSION) install MAKEOVERRIDES=

openfst-$(OPENFST_VERSION)/lib: | openfst-$(OPENFST_VERSION)/Makefile
	- cd openfst-$(OPENFST_VERSION) && \
	  [ -d lib64 ] && [ ! -d lib ] && ln -s lib64 lib

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

openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION)
	cd openfst-$(OPENFST_VERSION)/ && \
	./configure --prefix=`pwd` $(OPENFST_CONFIGURE) CXX="$(CXX)" \
		CXXFLAGS="$(CXXFLAGS) $(openfst_add_CXXFLAGS)" \
		LDFLAGS="$(LDFLAGS)" LIBS="-ldl"

openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
	  cp -p "$(DOWNLOAD_DIR)/openfst-$(OPENFST_VERSION).tar.gz" .; \
	else \
	  $(WGET) -nv -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
	  $(WGET) -nv -T 10 -t 3 -c https://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz; \
	fi

openfst_cleaned:
	-for d in openfst/ openfst-*/; do \
	   [ ! -f $$d/Makefile ] || $(MAKE) -C $$d clean; \
	done

#== SCTK =======================================================================

SCTK_CXFLAGS = -w -march=native
SCTK_MKENV = CFLAGS="$(CFLAGS) $(SCTK_CXFLAGS)" \
             CXXFLAGS="$(CXXFLAGS) -std=c++11 $(SCTK_CXFLAGS)" \

# Keep the existing target 'sclite' to avoid breaking the users who might have
# scripted it in.
.PHONY: sclite sctk_cleaned sctk_made

sclite sctk_made: sctk/.compiled

sctk/.compiled: sctk
	rm -f sctk/.compiled
	$(SCTK_MKENV) $(MAKE) -C sctk config
	$(SCTK_MKENV) $(MAKE) -C sctk all doc
	$(MAKE) -C sctk install
	touch sctk/.compiled

# The GitHub archive unpacks into SCTK-{40-character-long-hash}/
sctk: sctk-$(SCTK_GITHASH).tar.gz
	tar zxvf sctk-$(SCTK_GITHASH).tar.gz
	rm -rf sctk-$(SCTK_GITHASH) sctk
	mv SCTK-$(SCTK_GITHASH)* sctk-$(SCTK_GITHASH)
	ln -s sctk-$(SCTK_GITHASH) sctk
	touch sctk-$(SCTK_GITHASH).tar.gz

sctk-$(SCTK_GITHASH).tar.gz:
	if [ -d '$(DOWNLOAD_DIR)' ]; then \
	  cp -p '$(DOWNLOAD_DIR)/sctk-$(SCTK_GITHASH).tar.gz' .; \
	else \
	  $(WGET) -nv -T 10 -t 3 -O sctk-$(SCTK_GITHASH).tar.gz \
	    https://github.com/usnistgov/SCTK/archive/$(SCTK_GITHASH).tar.gz; \
	fi

sctk_cleaned:
	-for d in sctk/ sctk-*/; do \
	   [ ! -f $$d/.compiled ] || $(MAKE) -C $$d clean; \
	   rm -f $$d/.compiled; \
	done

#== sph2pipe ===================================================================

# Note the naming subtlety: the tarball extracts into a subdirectory
# 'sph2pipe-2.5', but over 9000 scripts call for 'sph2pipe_v2.5'.

sph2pipe: sph2pipe_v$(SPH2PIPE_VERSION)/sph2pipe
	rm -f sph2pipe
	ln -s sph2pipe_v$(SPH2PIPE_VERSION) sph2pipe
	touch -r $@ -c $^

sph2pipe_v$(SPH2PIPE_VERSION)/sph2pipe: sph2pipe_v$(SPH2PIPE_VERSION)/Makefile
	$(MAKE) -C sph2pipe_v$(SPH2PIPE_VERSION)

sph2pipe_v$(SPH2PIPE_VERSION)/Makefile: sph2pipe-$(SPH2PIPE_VERSION).tar.gz
	rm -rf sph2pipe_v*
	tar -xmzf sph2pipe-$(SPH2PIPE_VERSION).tar.gz
	mv sph2pipe-$(SPH2PIPE_VERSION) sph2pipe_v$(SPH2PIPE_VERSION)

sph2pipe-$(SPH2PIPE_VERSION).tar.gz:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
	  cp -p "$(DOWNLOAD_DIR)/sph2pipe-v$(SPH2PIPE_VERSION).tar.gz" \
	        sph2pipe-$(SPH2PIPE_VERSION).tar.gz; \
	else \
	  $(WGET) -nv -T 10 -t 3 -O sph2pipe-$(SPH2PIPE_VERSION).tar.gz \
	    https://github.com/burrmill/sph2pipe/archive/$(SPH2PIPE_VERSION).tar.gz; \
	fi

#== CUB ========================================================================

.PHONY: cub
cub: cub/common.mk

cub/common.mk: cub-$(CUB_VERSION).tar.gz
	rm -f cub
	tar -xmzf cub-$(CUB_VERSION).tar.gz
	ln -s cub-$(CUB_VERSION) cub

cub-$(CUB_VERSION).tar.gz:
	if [ -d "$(DOWNLOAD_DIR)" ]; then \
	  cp -p "$(DOWNLOAD_DIR)/cub-$(CUB_VERSION).tar.gz" .; \
	else \
	  $(WGET) -nv -T 10 -t 3 -O cub-$(CUB_VERSION).tar.gz \
	    https://github.com/NVlabs/cub/archive/$(CUB_VERSION).tar.gz; \
	fi

#== No OpenBLAS ================================================================

.PHONY: openblas
openblas:
	@echo "##===============================================##"
	@echo "##  WARNING: The 'openblas' target is obsolete.  ##"
	@echo "##  Run 'extras/install_openblas.sh' instead.    ##"
	@echo "##===============================================##"
	@false
