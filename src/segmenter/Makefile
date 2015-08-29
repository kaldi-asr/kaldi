all:

include ../kaldi.mk

TESTFILES = segmentation-io-test

OBJFILES = segmenter.o

LIBNAME = kaldi-segmenter

ADDLIBS = ../gmm/kaldi-gmm.a  \
	../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

