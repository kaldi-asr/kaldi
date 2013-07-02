# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

SHELL := /bin/bash


SUBDIRS = base matrix util feat tree thread gmm tied transform sgmm \
          fstext hmm lm decoder lat cudamatrix nnet \
          bin fstbin gmmbin fgmmbin tiedbin sgmmbin featbin \
          nnetbin latbin sgmm2 sgmm2bin nnet-cpu nnet-cpubin kwsbin
SUBDIRS_LIB = $(filter-out %bin, $(SUBDIRS))


# Optional subdirectories
EXT_SUBDIRS = online onlinebin  # python-kaldi-decoding
EXT_SUBDIRS_LIB = $(filter-out %bin, $(EXT_SUBDIRS))

include kaldi.mk

# Reset the default goal, so that the all target will become default
.DEFAULT_GOAL :=
all: test_install kaldi.mk $(SUBDIRS)
	-echo Done

mklibdir:
	mkdir -p $(KALDILIBDIR)

#I don't want to call rm -rf
rmlibdir:
	-rm $(KALDILIBDIR)/*{.so,.a,.o}
	-rmdir $(KALDILIBDIR)

biglib: $(SUBDIRS_LIB) 
ifneq ($(KALDI_FLAVOR), dynamic)
	$(warning the following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all)
endif
	-gcc -shared -o $(KALDILIBDIR)/libkaldi.so -Wl,-soname=libkaldi.so,--whole-archive  $(SUBDIRS_LIB:=/kaldi-*.a) -Wl,--no-whole-archive

biglibext: $(EXT_SUBDIRS_LIB)
ifneq ($(KALDI_FLAVOR), dynamic)
	$(warning The following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all.)
endif
	-gcc -shared -o $(KALDILIBDIR)/libkaldi_ext.so -Wl,-soname=libkaldi_ext.so,--whole-archive  $(EXT_SUBDIRS_LIB:=/kaldi-*.a) -Wl,--no-whole-archive

kaldi.mk:
	@[ -f kaldi.mk ] || { echo "kaldi.mk does not exist; you have to run ./configure"; exit 1; }

# Compile optional stuff 
ext: test_install ext_depend $(SUBDIRS) $(EXT_SUBDIRS)
	-echo Done

# Note: if you have put the OpenFst installation somewhere else, you can just
# delete or comment out the lines below.
OPENFST_VER = $(shell grep 'PACKAGE_VERSION' ../tools/openfst/Makefile | sed -e 's:.*= ::')
test_install:
	@[ "$(OPENFST_VER)" == '1.2.10' ] || [ "$(OPENFST_VER)" == '1.3.2' ] || { echo "You now need openfst-1.2.10.  Do: cd ../tools; svn update; ./install.sh; cd ../src; make depend; make"; exit 1; };

check_portaudio:
	@[ -d ../tools/portaudio ] || ( cd ../tools;  ./install_portaudio.sh )


clean: rmlibdir
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x clean; done

# the part after the expands to "base/test matrix/test util/test ...."
test: $(addsuffix /test, $(SUBDIRS))

ext_test: $(addsuffix /test, $(EXT_SUBDIRS))

# Define an implicit rule, expands to e.g.:
#  base/test: base
#     $(MAKE) -C base test 
%/test: % mklibdir
	$(MAKE) -C $< test

valgrind:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

depend: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done

ext_depend: check_portaudio
	-for x in $(EXT_SUBDIRS); do $(MAKE) -C $$x depend; done


.PHONY: $(SUBDIRS)
$(SUBDIRS) :
	$(MAKE) -C $@ 

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) :
	$(MAKE) -C $@


### Dependency list ###
# this is necessary for correct parallel compilation
#1)The tools depend on all the libraries
bin fstbin gmmbin fgmmbin tiedbin sgmmbin sgmm2bin featbin nnetbin nnet-cpubin latbin: base matrix util feat tree optimization thread gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet-cpu

#2)The libraries have inter-dependencies
base:
matrix : base
util: base matrix
thread: util
feat: base matrix util gmm transform
tree: base util matrix
optimization: base matrix
gmm: base util matrix tree
tied: base util matrix gmm tree transform
transform: base util matrix gmm tree
sgmm: base util matrix gmm tree transform thread
sgmm2: base util matrix gmm tree transform thread
fstext: base util matrix tree
hmm: base tree matrix 
lm: base util
decoder: base util matrix gmm sgmm hmm tree transform
lat: base util
cudamatrix: base util matrix	
nnet: base util matrix cudamatrix
nnet-cpu: base util matrix thread
#3)Dependencies for optional parts of Kaldi
onlinebin: base matrix util feat tree optimization gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet-cpu online
# python-kaldi-decoding: base matrix util feat tree optimization thread gmm tied transform sgmm sgmm2 fstext hmm decoder lat online
online: decoder
kwsbin: fstext lat base util
