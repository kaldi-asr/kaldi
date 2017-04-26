all:

include ../kaldi.mk

TESTFILES = segmentation-io-test

OBJFILES = segment.o segmentation.o segmentation-utils.o \
					 segmentation-post-processor.o 

LIBNAME = kaldi-segmenter

ADDLIBS = ../tree/kaldi-tree.a ../gmm/kaldi-gmm.a  \
	../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

