
all:

include ../kaldi.mk

# note, build-tree-utils-test also tests build-tree-questions.cc

TESTFILES = event-map-test context-dep-test build-tree-utils-test cluster-utils-test build-tree-test

OBJFILES = event-map.o context-dep.o clusterable-classes.o cluster-utils.o build-tree-utils.o build-tree.o build-tree-questions.o tree-renderer.o

LIBFILE = kaldi-tree.a

all:  $(LIBFILE) $(TESTFILES)

$(LIBFILE): $(OBJFILES)
	while ! mkdir .lock; do sleep 1; done
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)
	rmdir .lock


$(TESTFILES): $(LIBFILE) ../base/kaldi-base.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

