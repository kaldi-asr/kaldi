
all:

include ../kaldi.mk

# note, build-tree-utils-test also tests build-tree-questions.cc

TESTFILES = event-map-test context-dep-test build-tree-utils-test \
						cluster-utils-test build-tree-test


OBJFILES = event-map.o context-dep.o context-dep-multi.o clusterable-classes.o clusterable-classes-multi.o cluster-utils.o cluster-utils-multi.o \
					 build-tree-utils.o build-tree-utils-multi.o build-tree.o build-tree-multi.o build-tree-questions.o tree-renderer.o build-tree-virtual.o

LIBNAME = kaldi-tree
ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
