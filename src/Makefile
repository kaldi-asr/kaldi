# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories.

SHELL := /bin/bash

SUBDIRS = base matrix util feat cudafeat tree gmm transform \
          fstext hmm lm decoder lat kws cudamatrix \
          bin fstbin gmmbin fgmmbin featbin cudafeatbin \
          latbin nnet3 rnnlm chain nnet3bin kwsbin \
          ivector ivectorbin online2 online2bin lmbin chainbin rnnlmbin \
          cudadecoder cudadecoderbin cblasext

MEMTESTDIRS = base matrix util feat cudafeat tree gmm transform \
          fstext hmm lm decoder lat kws chain \
          bin fstbin gmmbin fgmmbin featbin cudafeatbin \
          latbin nnet3 rnnlm nnet3bin kwsbin \
          ivector ivectorbin online2 online2bin lmbin

CUDAMEMTESTDIR = cudamatrix

SUBDIRS_LIB = $(filter-out %bin, $(SUBDIRS))

KALDI_SONAME ?= libkaldi.so


include kaldi.mk

# Reset the default goal, so that the all target will become default
.DEFAULT_GOAL :=
all: $(SUBDIRS) matrix/test
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
	@echo ""
	@echo "The kaldi.mk file was generated using a different version of configure script. Please rerun the configure again"
	@test -f ./kaldi.mk && echo  "Hint: Previous configure command line: " && head -n 2 ./kaldi.mk | grep configure | sed 's/^# *//g'
	@echo ""
	@false
endif

biglib: $(SUBDIRS_LIB)
ifeq ($(KALDI_FLAVOR), dynamic)
ifeq ($(shell uname), Darwin)
	$(CXX) -dynamiclib -o $(KALDILIBDIR)/libkaldi.dylib -install_name @rpath/libkaldi.dylib -framework Accelerate $(LDFLAGS) $(SUBDIRS_LIB:=/*.dylib)
else
ifeq ($(shell uname), Linux)
	#$(warning the following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all)
	$(CXX) -shared -o $(KALDILIBDIR)/$(KALDI_SONAME) -Wl,-soname=$(KALDI_SONAME),--whole-archive  $(SUBDIRS_LIB:=/kaldi-*.a) $(LDLIBS) -Wl,--no-whole-archive
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


.PHONY: $(SUBDIRS)
$(SUBDIRS) : checkversion kaldi.mk mklibdir
	$(MAKE) -C $@

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) : checkversion kaldi.mk mklibdir ext_depend
	$(MAKE) -C $@


### Dependency list ###
# this is necessary for correct parallel compilation
#1)The tools depend on all the libraries

bin fstbin gmmbin fgmmbin featbin cudafeatbin nnet3bin chainbin latbin ivectorbin lmbin kwsbin online2bin rnnlmbin cudadecoderbin: \
 base matrix util feat cudafeat tree gmm transform fstext hmm \
 lm decoder lat cudadecoder cudamatrix nnet3 ivector chain kws online2 rnnlm

#2)The libraries have inter-dependencies
base: base/.depend.mk
cblasext: base
matrix: base cblasext
util: base matrix cblasext
feat: base cudamatrix matrix cblasext util gmm transform tree
tree: base util matrix cblasext
gmm: base util matrix cblasext tree
transform: base util matrix cblasext gmm tree
fstext: base util matrix cblasext tree
hmm: base tree matrix cblasext util
lm: base util matrix cblasext fstext
decoder: base util matrix cblasext gmm hmm tree transform lat fstext
lat: base util hmm tree matrix cblasext
cudamatrix: base util matrix cblasext
nnet3: base util matrix cblasext decoder lat gmm hmm tree transform cudamatrix chain fstext
rnnlm: base util matrix cblasext cudamatrix nnet3 lm hmm
chain: lat hmm tree fstext matrix cblasext cudamatrix util base
ivector: base util matrix cblasext transform tree gmm
cudafeat: base cudamatrix matrix cblasext util gmm transform tree feat cudamatrix online2
onlinebin: base cudamatrix matrix cblasext util feat tree gmm transform fstext hmm lm decoder lat cudamatrix online
# python-kaldi-decoding: base cudamatrix matrix cblasext util feat tree gmm transform fstext hmm decoder lat online

cudafeatbin: base cudamatrix matrix cblasext util gmm transform tree feat cudamatrix cudafeat online2
online: decoder gmm transform feat matrix cblasext util base lat hmm tree
online2: decoder gmm transform feat matrix cblasext util base lat hmm tree ivector cudamatrix nnet3 chain
kws: base util hmm tree matrix cblasext lat
cudadecoder:  cudamatrix cudafeat online2 nnet3 ivector feat fstext lat chain transform

