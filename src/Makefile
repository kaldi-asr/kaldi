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

MEMTESTDIRS = $(filter-out chainbin cudamatrix rnnlmbin $(SUBDIRS))
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

SUBDIRS_LIB = $(filter-out %bin, $(SUBDIRS))
SUBDIRS_BIN = $(filter     %bin, $(SUBDIRS))

KALDI_SONAME ?= libkaldi.so

# Reset the default goal, so that the all target will become default
.DEFAULT_GOAL :=
all: $(SUBDIRS) matrix/test
	-echo Done

mklibdir:
	test -d $(KALDILIBDIR) || mkdir $(KALDILIBDIR)

# Don't call rm -rf.
rmlibdir:
ifneq ($(KALDILIBDIR), )
	-rm $(KALDILIBDIR)/*{.so,.a,.o}
	-rmdir $(KALDILIBDIR)
else
	@true
endif

.PHONY: checkversion
checkversion:
ifeq ($(shell ./configure --version),$(CONFIGURE_VERSION))
	@echo "The version of configure script matches kaldi.mk version. Good."
else
	@echo ""
	@echo "The kaldi.mk file was generated using a different version of configure script. Please rerun the configure again"
	@test -f ./kaldi.mk && echo  "Hint: Previous configure command line: " && head -n 2 ./kaldi.mk | grep configure | sed 's/^# *//g'
	@echo ""
	@false
endif

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

kaldi.mk:
	@[ -f kaldi.mk ] || { echo "kaldi.mk does not exist; you have to run ./configure"; exit 1; }

# Compile optional stuff
ext: ext_depend $(SUBDIRS) $(EXT_SUBDIRS)
	-echo Done

check_portaudio:
	@[ -d ../tools/portaudio ] || ( cd ../tools;  ./install_portaudio.sh )

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

cudavalgrind:
	-for x in $(CUDAMEMTESTDIR); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

valgrind:
	-for x in $(MEMTESTDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

base/.depend.mk:
	$(MAKE) depend

depend: $(addsuffix /depend, $(SUBDIRS))

%/depend:
	$(MAKE) -C $(dir $@) depend


ext_depend: check_portaudio
	-for x in $(EXT_SUBDIRS); do $(MAKE) -C $$x depend; done

# 'make libs' to skip binaries, and build only libraries.
.PHONY: libs $(SUBDIRS_LIB)
libs: $(SUBDIRS_LIB) ;
$(SUBDIRS_LIB) : checkversion kaldi.mk mklibdir
	$(MAKE) -C $@

.PHONY: bins $(SUBDIRS_BIN) libs
$(SUBDIRS_BIN) : checkversion kaldi.mk mklibdir
	$(MAKE) -C $@

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) : checkversion kaldi.mk mklibdir ext_depend
	$(MAKE) -C $@

### Dependency list ###
# Three base libraries that mostly everything depends on
BMU = base matrix util

# this is necessary for correct parallel compilation
#1) The tools depend on all the libraries (which is a nonsense)
bin chainbin featbin fgmmbin fstbin gmmbin ivectorbin \
    kwsbin latbin lmbin nnet2bin nnet3bin nnetbin online2bin rnnlmbin : \
 $(BMU) chain cudamatrix decoder feat fstext gmm hmm ivector \
        kws lat lm nnet nnet2 nnet3 online2 rnnlm transform tree

#2) The libraries have inter-dependencies
base:   base/.depend.mk
matrix: base
util:   base matrix
feat:      $(BMU) gmm transform tree
tree:      $(BMU)
gmm:       $(BMU) tree
transform: $(BMU) gmm tree
sgmm2:     $(BMU) gmm hmm transform tree
fstext:    $(BMU) tree
hmm:       $(BMU) tree
lm:        $(BMU) fstext
decoder:   $(BMU) fstext gmm hmm lat transform tree
lat:       $(BMU) hmm tree
cudamatrix: $(BMU)
nnet:      $(BMU) cudamatrix hmm tree
nnet2:     $(BMU) cudamatrix gmm hmm lat transform tree
nnet3:     $(BMU) chain cudamatrix decoder fstext gmm hmm lat transform tree
rnnlm:     $(BMU) cudamatrix hmm lm nnet3
chain:     $(BMU) cudamatrix fstext hmm lat tree
ivector:   $(BMU) gmm transform tree
#3) Dependencies for optional parts of Kaldi
onlinebin: $(BMU) cudamatrix decoder feat fstext gmm hmm lat lm nnet nnet2 online sgmm2 transform tree
# python-kaldi-decoding: $(BMU) decoder feat fstext gmm hmm lat online sgmm2 transform tree
online:    $(BMU) decoder feat gmm hmm lat transform tree
online2:   $(BMU) chain cudamatrix decoder feat gmm hmm ivector lat nnet2 nnet3 transform tree
kws:       $(BMU) hmm lat tree
#4) CUDA decoder library and binary dependencies.
cudafeat:  $(BMU) cudamatrix feat gmm online2 transform tree
cudafeatbin: cudafeat
cudadecoder: chain cudafeat cudamatrix feat fstext ivector lat nnet3 online2 transform
cudadecoderbin: cudadecoder cudafeat
