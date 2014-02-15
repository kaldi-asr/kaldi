
all:

include ../kaldi.mk


BINFILES = online2-wav-gmm-latgen-faster apply-cmvn-online

OBJFILES = 

TESTFILES =

ADDLIBS = ../online2/kaldi-online2.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../thread/kaldi-thread.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
