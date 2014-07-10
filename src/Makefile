# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

SHELL := /bin/bash


SUBDIRS = base matrix util feat tree thread gmm transform sgmm \
          fstext hmm lm decoder lat cudamatrix nnet \
          bin fstbin gmmbin fgmmbin sgmmbin featbin \
          nnetbin latbin sgmm2 sgmm2bin nnet2 nnet2bin kwsbin \
          ivector ivectorbin

MEMTESTDIRS = base matrix util feat tree thread gmm transform sgmm \
          fstext hmm lm decoder lat nnet \
          bin fstbin gmmbin fgmmbin sgmmbin featbin \
          nnetbin latbin sgmm2 nnet2 nnet2bin sgmm2bin kwsbin \
          ivector ivectorbin

CUDAMEMTESTDIR = cudamatrix

SUBDIRS_LIB = $(filter-out %bin, $(SUBDIRS))


# Optional subdirectories
EXT_SUBDIRS = online onlinebin  # python-kaldi-decoding
EXT_SUBDIRS_LIB = $(filter-out %bin, $(EXT_SUBDIRS))

include kaldi.mk

# Reset the default goal, so that the all target will become default
.DEFAULT_GOAL :=
all: checkversion test_dependencies kaldi.mk mklibdir $(SUBDIRS)
	-echo Done

mklibdir:
	test -d $(KALDILIBDIR) || mkdir $(KALDILIBDIR)

#I don't want to call rm -rf
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
	$(error The kaldi.mk file was generated using a different version of configure script. Run the configure script again.)
endif

biglib: $(SUBDIRS_LIB) 
ifeq ($(KALDI_FLAVOR), dynamic)
ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o $(KALDILIBDIR)/libkaldi.dylib -install_name @rpath/libkaldi.dylib -framework Accelerate $(LDFLAGS) $(SUBDIRS_LIB:=/*.dylib)
else
ifeq ($(shell uname), Linux)
	#$(warning the following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all)
	$(CXX) -shared -o $(KALDILIBDIR)/libkaldi.so -Wl,-soname=libkaldi.so,--whole-archive  $(SUBDIRS_LIB:=/kaldi-*.a) -Wl,--no-whole-archive
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
	#$(warning The following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all.)
	$(CXX) -shared -o $(KALDILIBDIR)/libkaldi_ext.so -Wl,-soname=libkaldi_ext.so,--whole-archive  $(EXT_SUBDIRS_LIB:=/kaldi-*.a) -Wl,--no-whole-archive
else
	$(error Dynamic libraries not supported on this platform. Run configure with --static flag. )
endif
endif
endif

kaldi.mk:
	@[ -f kaldi.mk ] || { echo "kaldi.mk does not exist; you have to run ./configure"; exit 1; }

# Compile optional stuff 
ext: test_dependencies ext_depend $(SUBDIRS) $(EXT_SUBDIRS)
	-echo Done

# Note: if you have put the OpenFst installation somewhere else, you can just
# delete or comment out the lines below.
OPENFST_VER = $(shell grep 'PACKAGE_VERSION' ../tools/openfst/Makefile | sed -e 's:.*= ::')
test_dependencies:
	[ "$(OPENFST_VER)" == '1.3.2' ] || [ "$(OPENFST_VER)" == '1.3.3' ] || [ "$(OPENFST_VER)" == '1.3.4' ] || { echo "You now need openfst-1.3.2 or later. cd ../tools; svn update; ./install.sh; cd ../src; make depend; make"; exit 1; };

check_portaudio:
	@[ -d ../tools/portaudio ] || ( cd ../tools;  ./install_portaudio.sh )


clean: rmlibdir
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x clean; done

test: $(addsuffix /test, $(SUBDIRS))

ext_test: $(addsuffix /test, $(EXT_SUBDIRS))

# Define an implicit rule, expands to e.g.:
#  base/test: base
#     $(MAKE) -C base test 
%/test: % mklibdir
	$(MAKE) -C $< test

cudavalgrind:
	-for x in $(CUDAMEMTESTDIR); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

valgrind:
	-for x in $(MEMTESTDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done


depend: $(addsuffix /depend, $(SUBDIRS))

%/depend:
	$(MAKE) -C $(dir $@) depend


ext_depend: check_portaudio
	-for x in $(EXT_SUBDIRS); do $(MAKE) -C $$x depend; done


.PHONY: $(SUBDIRS)
$(SUBDIRS) : mklibdir
	$(MAKE) -C $@ 

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) : mklibdir
	$(MAKE) -C $@


### Dependency list ###
# this is necessary for correct parallel compilation
#1)The tools depend on all the libraries

bin fstbin gmmbin fgmmbin sgmmbin sgmm2bin featbin nnetbin nnet2bin latbin ivectorbin: \
 base matrix util feat tree optimization thread gmm transform sgmm sgmm2 fstext hmm \
 lm decoder lat cudamatrix nnet nnet2 ivector

bin fstbin gmmbin fgmmbin sgmmbin sgmm2bin featbin nnetbin nnet2bin latbin ivectorbin: \
 base matrix util feat tree optimization thread gmm transform sgmm sgmm2 fstext hmm \
 lm decoder lat cudamatrix nnet nnet2 ivector

#2)The libraries have inter-dependencies
base:
matrix : base
util: base matrix
thread: util
feat: base matrix util gmm transform
tree: base util matrix
optimization: base matrix
gmm: base util matrix tree thread
transform: base util matrix gmm tree
sgmm: base util matrix gmm tree transform thread hmm
sgmm2: base util matrix gmm tree transform thread hmm
fstext: base util matrix tree
hmm: base tree matrix 
lm: base util
decoder: base util matrix gmm sgmm hmm tree transform lat
lat: base util hmm
cudamatrix: base util matrix	
nnet: base util matrix cudamatrix
nnet2: base util matrix thread lat
ivector: base util matrix thread transform tree gmm 
#3)Dependencies for optional parts of Kaldi
onlinebin: base matrix util feat tree optimization gmm transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet2 online
# python-kaldi-decoding: base matrix util feat tree optimization thread gmm transform sgmm sgmm2 fstext hmm decoder lat online
online: decoder
kwsbin: fstext lat base util
