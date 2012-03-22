# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

SHELL := /bin/bash

# Compile subdirectories starting with the things "depended on".
SUBDIRS = base matrix util feat tree optimization gmm tied transform sgmm \
          fstext hmm lm decoder lat \
          bin fstbin gmmbin fgmmbin tiedbin sgmmbin featbin \
          nnet_cpu nnetbin_cpu latbin rnn

all: test_install remove_locks $(SUBDIRS) 
	echo Done

OPENFST_VER = $(shell grep 'PACKAGE_VERSION' $(FSTROOT)/Makefile | sed -e 's:.*= ::')
test_install:
	[ $(OPENFST_VER) == '1.2.10' ] || ( echo "You now need openfst-1.2.10.  Do: cd ../tools; svn update; ./install.sh; cd ../src; make depend; make"; exit 1 )

remove_locks:
	rmdir */.lock 2>/dev/null || exit 0;

clean: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done

# When testing, stop on first failure.
test:
	for x in $(SUBDIRS); do $(MAKE) -C $$x test || (echo "test $$x failed"; exit 1); done  

valgrind:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x valgrind || (echo "valgrind on $$x failed"; exit 1); done

depend: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done

$(SUBDIRS) : test_install remove_locks .phony
	$(MAKE) -C $@ 


.phony:


#FSTROOT = /exports/home/aghoshal/work/tools/src/kaldi/trunk/tools/openfst
FSTROOT = ../tools/openfst
