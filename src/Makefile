# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

# Compile subdirectories starting with the things "depended on".

SUBDIRS = base matrix util feat tree optimization gmm transform sgmm \
          fstext hmm lm decoder \
          bin fstbin gmmbin fgmmbin sgmmbin featbin \
          nnet nnetbin cudamatrix cudannet cudannetbin lat latbin

all: $(SUBDIRS) 
	echo Done

clean: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done

# When testing, stop on first failure.
test:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x test || (echo "test $$x failed"; exit 1); done  

valgrind:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x valgrind || (echo "valgrind on $$x failed"; exit 1); done

depend: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done

$(SUBDIRS) : .phony
	$(MAKE) -C $@ 


.phony:


