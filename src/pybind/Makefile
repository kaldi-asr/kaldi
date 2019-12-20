
# make "all" the target.
all:


# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk


ifeq ($(KALDI_FLAVOR),static)
  $(error You cannot build the pybind directory unless the build was dynamic; reconfigure with --shared option.)
endif

PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes)

CXXFLAGS += $(PYBIND_INCLUDES)

PYBIND_EXTENSION=$(shell python3-config --extension-suffix)
LIBFILE=kaldi-pybind
LIBFILE_EXTENSION=$(PYBIND_EXTENSION)

# pybind11 is heavily templated and generates code that is bloated before optimization.
# -flto is link time optimization which apparently is important.
CXXFLAGS += -O3 -flto -I.
LDFLAGS += -flto
# TODO:

ifeq ($(shell uname),Darwin)
  LDFLAGS += -undefined dynamic_lookup
endif

CCFILES = \
chain/chain_pybind.cc \
chain/chain_supervision_pybind.cc \
feat/feat_pybind.cc \
feat/feature_pybind.cc \
feat/wave_reader_pybind.cc \
fst/arc_pybind.cc \
fst/compile_pybind.cc \
fst/fst_pybind.cc \
fst/symbol_table_pybind.cc \
fst/vector_fst_pybind.cc \
fst/weight_pybind.cc \
fstext/kaldi_fst_io_pybind.cc \
kaldi_pybind.cc \
matrix/matrix_common_pybind.cc \
matrix/matrix_pybind.cc \
matrix/vector_pybind.cc \
nnet3/nnet3_pybind.cc \
nnet3/nnet_chain_example_pybind.cc \
nnet3/nnet_common_pybind.cc \
nnet3/nnet_example_pybind.cc \
util/table_types_pybind.cc

CCFILES_OBJS := $(CCFILES:%.cc=%.o)

LIBNAME = kaldi_pybind


ADDLIBS = \
../base/kaldi-base.a \
../chain/kaldi-chain.a \
../feat/kaldi-feat.a \
../fstext/kaldi-fstext.a \
../matrix/kaldi-matrix.a \
../nnet3/kaldi-nnet3.a \
../util/kaldi-util.a


EXTRA_LDLIBS += $(foreach dep,$(ADDLIBS), $(dir $(dep))lib$(notdir $(basename $(dep))).so)
EXTRA_LDLIBS += ../../tools/openfst/lib/libfstscript.so

LDFLAGS += -Wl,-rpath=$(CURDIR)/../../tools/openfst/lib

LIBFILE=$(LIBNAME)$(LIBFILE_EXTENSION)


.PHONY: all clean test

all: $(LIBFILE)

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $^

$(LIBFILE): $(ADDLIBS) $(CCFILES_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(CCFILES_OBJS) -Wl,--no-whole-archive -Wl,-rpath=$(CURDIR)/../lib $(LDFLAGS) $(LDLIBS) $(EXTRA_LDLIBS)
	python3 -c 'import kaldi_pybind'  # this line is a test.

clean:
	-rm -f *.so
	-rm -rf __pycache__
	-rm -f $(CCFILES_OBJS)

test: all
	python3 tests/test_kaldi_pybind.py
	python3 tests/test_matrix.py
	python3 tests/test_wave.py
	python3 tests/test_feat.py
	make -C fst test
	make -C chain test
	make -C nnet3 test

# valgrind-python.supp is from http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp
# since we do not compile Python from souce, we follow the comment in valgrind-python.supp
# to uncomment suppressions for PyObject_Free and PyObject_Realloc.
valgrind:
	valgrind --tool=memcheck --suppressions=./valgrind-python.supp \
   python3 -E -tt ./tests/test_matrix.py
