
all:

SWIG_PYTHON_BINDING_NAME = online2_nnet3_latgen_i2x_wrapper
EXTRA_CXXFLAGS += -I/usr/include/python3.5

$(SWIG_PYTHON_BINDING_NAME)_wrap.cc: $(SWIG_PYTHON_BINDING_NAME).i
	../../tools/swig/build/bin/swig -c++ -python $(SWIG_PYTHON_BINDING_NAME).i
	mv $(SWIG_PYTHON_BINDING_NAME)_wrap.cxx $(SWIG_PYTHON_BINDING_NAME)_wrap.cc

include ../kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =

OBJFILES = $(SWIG_PYTHON_BINDING_NAME)_wrap.o

ADDLIBS = ../online2/kaldi-online2.a \
          ../ivector/kaldi-ivector.a ../nnet3/kaldi-nnet3.a \
          ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
          ../fstext/kaldi-fstext.a


include ../makefiles/default_rules.mk

all: _$(SWIG_PYTHON_BINDING_NAME).so

_$(SWIG_PYTHON_BINDING_NAME).so: $(OBJFILES) $(ADDLIBS)
	$(CXX) -shared -o $@ $(OBJFILES) $(LDFLAGS) $(ADDLIBS) $(LDLIBS)

