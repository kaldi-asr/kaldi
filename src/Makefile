# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories.

SHELL := /bin/bash

# Please keep sorted alphabetically, and start each with a letter which
# is different from the first one in the last word one the row above it.
SUBDIRS := base bin chain chainbin cudamatrix decoder		\
           feat featbin fgmmbin fstbin fstext			\
           gmm gmmbin hmm                               	\
           ivector ivectorbin kws kwsbin	        	\
           lat latbin lm lmbin matrix				\
           nnet nnetbin nnet2 nnet2bin nnet3 nnet3bin		\
           online2 online2bin rnnlm rnnlmbin            	\
           transform tree util

# Will also build these if configured with --with-cudadecoder
# (default 'true' if CUDA is used, else 'false')
CUDADECODER = cudafeat cudafeatbin cudadecoder cudadecoderbin

MEMTESTDIRS = $(filter-out chainbin cudamatrix rnnlmbin, $(SUBDIRS))
CUDAMEMTESTDIR = cudamatrix

# Optional subdirectories
EXT_SUBDIRS := online onlinebin sgmm2 sgmm2bin  # python-kaldi-decoding
EXT_SUBDIRS_LIB = $(filter-out %bin, $(EXT_SUBDIRS))

include kaldi.mk

ifeq ($(CUDA), true)
ifeq ($(WITH_CUDADECODER), true)
SUBDIRS += $(CUDADECODER)
endif
endif

ifeq ($(ROCM), true)
ifeq ($(WITH_CUDADECODER), true)
SUBDIRS += $(CUDADECODER)
endif
endif

SUBDIRS_LIB = $(filter-out %bin, $(SUBDIRS))
SUBDIRS_BIN = $(filter     %bin, $(SUBDIRS))

KALDI_SONAME ?= libkaldi.so

# Reset the default goal, so that the all target will become default
.DEFAULT_GOAL :=
.PHONY: all mklibdir rmlibdir checkversion
all: $(SUBDIRS) matrix/test
	@echo Done

ifneq ($(KALDILIBDIR), )
mklibdir: | $(KALDILIBDIR)/

$(KALDILIBDIR)/:
	mkdir $(KALDILIBDIR)
else
mklibdir: ;
endif

# Don't call rm -rf.
rmlibdir:
ifeq ($(KALDI_FLAVOR), dynamic)
ifneq ($(KALDILIBDIR), )
	-rm -f $(KALDILIBDIR)/*{.so,.a,.o,.dylib}
	-rmdir 2>/dev/null $(KALDILIBDIR); true
else
# KALDILIBDIR might have been unset because of reconfigure. Do a best guess.
	@echo "Something seems wrong. Please re-run configure."
	@echo "I will continue but the cleanup might not be complete."
endif
endif

kaldi.mk:
	@echo "ERROR: kaldi.mk does not exist; run ./configure first.";
	@false

checkversion: | kaldi.mk
ifeq ($(shell ./configure --version),$(CONFIGURE_VERSION))
	@echo "The version of configure script matches kaldi.mk version. Good."
else
	@echo "ERROR: The kaldi.mk file was generated using a different" \
	      "version of configure script. Please run ./configure again."
	@echo "Hint: Previous configure command line: "
	@echo ""
	@head -n 2 ./kaldi.mk | grep configure | sed 's/^# */  /g'
	@echo ""
	@false
endif

.PHONY: biglib
biglib: $(SUBDIRS_LIB)
ifeq ($(KALDI_FLAVOR), dynamic)
ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o $(KALDILIBDIR)/libkaldi.dylib -install_name @rpath/libkaldi.dylib -framework Accelerate $(LDFLAGS) $(wildcard $(SUBDIRS_LIB:=/*.dylib))
else
ifeq ($(shell uname), Linux)
#	$(warning the following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all)
	$(CXX) -shared -o $(KALDILIBDIR)/$(KALDI_SONAME) -Wl,-soname=$(KALDI_SONAME),--whole-archive  $(wildcard $(SUBDIRS_LIB:=/kaldi-*.a)) -Wl,--no-whole-archive  $(LDLIBS)
else
	$(error Dynamic libraries not supported on this platform. Run configure with --static flag. )
endif
endif
endif

.PHONY: biglibext
biglibext: $(EXT_SUBDIRS_LIB)
ifeq ($(KALDI_FLAVOR), dynamic)
ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o $(KALDILIBDIR)/libkaldi_ext.dylib -install_name @rpath/libkaldi_ext.dylib -framework Accelerate $(LDFLAGS) $(EXT_SUBDIRS_LIB:=/*.dylib)
else
ifeq ($(shell uname), Linux)
#	$(warning The following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all.)
	$(CXX) -shared -o $(KALDILIBDIR)/libkaldi_ext.so -Wl,-soname=libkaldi_ext.so,--whole-archive  $(EXT_SUBDIRS_LIB:=/kaldi-*.a) -Wl,--no-whole-archive
else
	$(error Dynamic libraries not supported on this platform. Run configure with --static flag. )
endif
endif
endif

# Compile optional stuff
.PHONY: ext check_portaudio clean distclean test ext_test
ext: check_portaudio ext_depend $(SUBDIRS) $(EXT_SUBDIRS)
	@echo Done

check_portaudio: | ../tools/portaudio

../tools/portaudio:
	@echo "ERROR: portaudio is required. Run tools/install_portaudio.sh."
	@false

clean: rmlibdir
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x clean; done

distclean: clean
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x distclean; done

test: $(addsuffix /test, $(SUBDIRS_LIB))

ext_test: $(addsuffix /test, $(EXT_SUBDIRS_LIB))

# Define an implicit rule, expands to e.g.:
#  base/test: base
#     $(MAKE) -C base test
%/test: % mklibdir
	$(MAKE) -C $< test


.PHONY: cudavalgrind valgrind depend ext_depend
cudavalgrind:
	-for x in $(CUDAMEMTESTDIR); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

valgrind:
	-for x in $(MEMTESTDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

base/.depend.mk:
	$(MAKE) depend

depend: $(addsuffix /depend, $(SUBDIRS))

ext_depend: check_portaudio $(addsuffix /depend, $(EXT_SUBDIRS))

%/depend:
	$(MAKE) -C $(dir $@) depend

# 'make libs' to skip binaries, and build only libraries.
.PHONY: libs $(SUBDIRS_LIB)
libs: $(SUBDIRS_LIB) ;
$(SUBDIRS_LIB) : checkversion mklibdir
	$(MAKE) -C $@

.PHONY: bins $(SUBDIRS_BIN) libs
$(SUBDIRS_BIN) : checkversion mklibdir
	$(MAKE) -C $@

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) : checkversion mklibdir ext_depend
	$(MAKE) -C $@

########################################################################
### Dependency list

# 1) Binaries. A bin's main lib is listed first, extra space, then the rest of
#    dependencies in lexicographic order.
#    Exceptions: fgmmbin's main lib is gmm; bin does not have one.
bin:        decoder  fstext gmm hmm lat lm transform tree
chainbin:   chain  nnet3
featbin:    feat  hmm
fgmmbin:    gmm  decoder feat hmm lat transform
fstbin:     fstext  decoder
gmmbin:     gmm  decoder feat fstext hmm lat transform
ivectorbin: ivector  hmm
kwsbin:     kws
latbin:     lat  cudamatrix fstext lm nnet3 rnnlm
lmbin:      lm
nnet2bin:   nnet2  decoder lat nnet fstext
nnet3bin:   nnet3 online2
nnetbin:    nnet  hmm lat
rnnlmbin:   rnnlm  decoder fstext lat tree

# 2.  The library inter-dependencies.
# 2a. Low-level libs. Normally these are those which do not have a corresponding
#     LIBNAMEbin/ directory, but this is not universal. These 5 libraries are
#     guaranteed to have the following unbroken chain of dependencies:
#          { cudamatrix | tree } > util > matrix > base.

base:   base/.depend.mk
matrix: base
util:   matrix
cudamatrix tree: util

# 2b. Higher-level libs.
#     All libraries must depend on the "big three": base, matrix, util.
#     Dependency on cudamatrix, tree, or util transitively creates a dependency
#     on matrix and base, so either of these three is both enough and necessary.
#
#     Relying on other implicit dependencies is *discouraged*: for example,
#     if you depend on tree and gmm, do *not* use gmm alone as a proxy to
#     create the common dependencies listed above, but rather explicitly list
#     both gmm and tree. Otherwise, you are risking breaking a long chain of
#     dependencies by removing an unused one.
feat:      gmm transform tree
gmm:       tree
transform: gmm tree
fstext:    tree
hmm:       tree
kws:       hmm lat tree
lm:        fstext util
decoder:   fstext gmm hmm lat transform tree
lat:       hmm tree
nnet:      cudamatrix hmm tree
nnet2:     cudamatrix gmm hmm lat transform tree
nnet3:     chain cudamatrix decoder fstext gmm hmm lat transform tree
rnnlm:     cudamatrix hmm lm nnet3
chain:     cudamatrix fstext hmm lat tree
ivector:   gmm transform tree

#3) Dependencies for optional parts of Kaldi.
onlinebin: online
online:    cudamatrix decoder feat fstext gmm hmm lat lm nnet nnet2 online sgmm2 transform tree
online2bin: online2 fstext
online2:   chain cudamatrix decoder feat gmm hmm ivector lat nnet2 nnet3 transform tree
# python-kaldi-decoding: decoder feat fstext gmm hmm lat online sgmm2 transform tree
sgmm2bin:  sgmm2 decoder feat fstext hmm lat
sgmm2:     gmm hmm transform tree

#4) CUDA decoder library and binary dependencies.
cudafeat:  cudamatrix feat gmm ivector online2
cudafeatbin: cudafeat decoder hmm transform tree
cudadecoder: cudafeat cudamatrix fstext gmm hmm lat matrix nnet3 online2 transform tree util
cudadecoderbin: cudadecoder chain decoder feat ivector nnet2
