kaldi=../../../../..
all:

include $(kaldi)/src/kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare
EXTRA_CXXFLAGS += -I$(kaldi)/src
BINFILES = latgen-faster-mapped-fake-am

OBJFILES =

TESTFILES =

ADDLIBS = $(kaldi)/src/decoder/kaldi-decoder.a $(kaldi)/src/lat/kaldi-lat.a $(kaldi)/src/lm/kaldi-lm.a \
          $(kaldi)/src/fstext/kaldi-fstext.a $(kaldi)/src/hmm/kaldi-hmm.a \
          $(kaldi)/src/transform/kaldi-transform.a $(kaldi)/src/gmm/kaldi-gmm.a \
          $(kaldi)/src/tree/kaldi-tree.a $(kaldi)/src/util/kaldi-util.a $(kaldi)/src/matrix/kaldi-matrix.a \
          $(kaldi)/src/base/kaldi-base.a


include $(kaldi)/src/makefiles/default_rules.mk

