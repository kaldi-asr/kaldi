# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

SHELL := /bin/bash

SUBDIRS = base matrix util feat tree gmm tied transform sgmm \
          fstext hmm lm decoder lat cudamatrix nnet nnet_cpu \
          bin fstbin gmmbin fgmmbin tiedbin sgmmbin featbin \
          nnetbin nnetbin_cpu latbin sgmm2 sgmm2bin

# Optional subdirectories
EXT_SUBDIRS = online onlinebin

all: test_install $(SUBDIRS) 
	echo Done

# Compile optional stuff 
ext: test_install check_portaudio $(SUBDIRS) $(EXT_SUBDIRS)

# Note: if you have put the OpenFst installation somewhere else, you can just
# delete or comment out the lines below.
OPENFST_VER = $(shell grep 'PACKAGE_VERSION' ../tools/openfst/Makefile | sed -e 's:.*= ::')
test_install:
	@[ $(OPENFST_VER) == '1.2.10' ] || { echo "You now need openfst-1.2.10.  Do: cd ../tools; svn update; ./install.sh; cd ../src; make depend; make"; exit 1; };

check_portaudio:
	@[ -d ../tools/portaudio ] || ( cd ../tools;  ./install_portaudio.sh )

clean: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done

# When testing, stop on first failure.
test:
	for x in $(SUBDIRS); do $(MAKE) -C $$x test || { echo "test $$x failed"; exit 1; }; done  

valgrind:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

depend: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done


.PHONY: $(SUBDIRS)
$(SUBDIRS) :
	$(MAKE) -C $@ 

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) :
	$(MAKE) -C $@


### Dependency list ###
# this is necessary for correct parallel compilation
#1)The tools depend on all the libraries
bin fstbin gmmbin fgmmbin tiedbin sgmmbin sgmm2bin featbin nnetbin nnetbin_cpu latbin: base matrix util feat tree optimization gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet_cpu rnn
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
nnet_cpu: base util matrix
rnn: base util matrix lat
#3)Dependencies for optional parts of Kaldi
onlinebin: base matrix util feat tree optimization gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet_cpu rnn online
online: decoder
