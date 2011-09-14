all:

include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare

BINFILES = rnn-rescore

OBJFILES = rnnlm.o

LIBFILE = rnnlm.a

all:  $(LIBFILE) $(BINFILES)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)

$(BINFILES): ../lat/kaldi-lat.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a ../rnn/rnnlm.a

# Rule below would expand to, e.g.:
# # ../base/kaldi-base.a:
# #       make -c ../base kaldi-base.a
# # -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(BINFILES)

depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk
