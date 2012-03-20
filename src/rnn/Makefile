all:

include ../kaldi.mk

BINFILES = rnn-rescore rnn-rescore-kaldi

OBJFILES = rnnlm.o

LIBFILE = rnnlm.a

all:  $(LIBFILE) $(BINFILES)


$(LIBFILE): $(OBJFILES)
	while ! mkdir .lock; do sleep 1; done
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)
	rmdir .lock

$(BINFILES): $(LIBFILE) ../lat/kaldi-lat.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

# Rule below would expand to, e.g.:
# # ../base/kaldi-base.a:
# #       make -c ../base kaldi-base.a
# # -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES) tmp* *.tmp

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk
