
# make "all" the target.
all:


# Disable linking math libs because not needed here. Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk


ifeq ($(KALDI_FLAVOR),static)
  $(error You cannot build the pybind directory unless the build was dynamic; reconfigure with --shared option.)
endif

PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes)

CXXFLAGS += $(PYBIND_INCLUDES)

PYBIND_EXTENSION := $(shell python3-config --extension-suffix)
LIBFILE := kaldi-pybind
LIBFILE_EXTENSION := $(PYBIND_EXTENSION)

# pybind11 is heavily templated and generates code that is bloated before optimization.
# -flto is link time optimization which apparently is important.
CXXFLAGS += -O3 -flto -I.
LDFLAGS += -flto

ifeq ($(shell uname),Darwin)
  LDFLAGS += -undefined dynamic_lookup
endif

# files should be sorted in alphabetic order
CCFILES = \
chain/chain_den_graph_pybind.cc \
chain/chain_pybind.cc \
chain/chain_supervision_pybind.cc \
chain/chain_training_pybind.cc \
cudamatrix/cu_device_pybind.cc \
cudamatrix/cu_matrix_pybind.cc \
cudamatrix/cu_vector_pybind.cc \
cudamatrix/cudamatrix_pybind.cc \
dlpack/dlpack_deleter.cc \
dlpack/dlpack_pybind.cc \
dlpack/dlpack_submatrix.cc \
dlpack/dlpack_subvector.cc \
feat/feat_pybind.cc \
feat/feature_pybind.cc \
feat/online_feature_pybind.cc \
feat/wave_reader_pybind.cc \
fst/arc_pybind.cc \
fst/compile_pybind.cc \
fst/fst_pybind.cc \
fst/symbol_table_pybind.cc \
fst/vector_fst_pybind.cc \
fst/weight_pybind.cc \
fstext/kaldi_fst_io_pybind.cc \
kaldi_pybind.cc \
matrix/compressed_matrix_pybind.cc \
matrix/kaldi_matrix_pybind.cc \
matrix/kaldi_vector_pybind.cc \
matrix/matrix_common_pybind.cc \
matrix/matrix_pybind.cc \
matrix/sparse_matrix_pybind.cc \
nnet3/nnet3_pybind.cc \
nnet3/nnet_chain_example_pybind.cc \
nnet3/nnet_common_pybind.cc \
nnet3/nnet_example_pybind.cc \
tests/test_dlpack_subvector.cc \
util/kaldi_io_pybind.cc \
util/table_types_pybind.cc \
util/util_pybind.cc

CCFILES_OBJS := $(CCFILES:%.cc=%.o)

LIBNAME := kaldi_pybind

# libs should be sorted in alphabetic order
ADDLIBS := \
../base/kaldi-base.a \
../chain/kaldi-chain.a \
../cudamatrix/kaldi-cudamatrix.a \
../feat/kaldi-feat.a \
../fstext/kaldi-fstext.a \
../matrix/kaldi-matrix.a \
../nnet3/kaldi-nnet3.a \
../util/kaldi-util.a


EXTRA_LDLIBS += $(foreach dep,$(ADDLIBS), $(dir $(dep))lib$(notdir $(basename $(dep))).so)
EXTRA_LDLIBS += ../../tools/openfst/lib/libfstscript.so

LDFLAGS += -Wl,-rpath=$(CURDIR)/../../tools/openfst/lib

LIBFILE=$(LIBNAME)$(LIBFILE_EXTENSION)

ifdef MKL_LD_PRELOAD
LD_PRELOAD += $(MKL_LD_PRELOAD)
export LD_PRELOAD
endif

.PHONY: all clean test

all: $(LIBFILE)

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(LIBFILE): $(ADDLIBS) $(CCFILES_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(CCFILES_OBJS) -Wl,--no-whole-archive -Wl,-rpath=$(CURDIR)/../lib $(LDFLAGS) $(LDLIBS) $(EXTRA_LDLIBS)
	python3 -c 'import kaldi_pybind'  # this line is a test.

clean:
	-rm -f *.so
	-rm -rf __pycache__
	-rm -f $(CCFILES_OBJS)
	-rm -f .depend.mk

test: all
	python3 tests/test_dlpack_subvector.py
	python3 tests/test_kaldi_pybind.py
	make -C chain test
	make -C cudamatrix test
	make -C dlpack test
	make -C feat test
	make -C fst test
	make -C matrix test
	make -C nnet3 test

# valgrind-python.supp is from http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp
# since we do not compile Python from source, we follow the comment in valgrind-python.supp
# to uncomment suppressions for PyObject_Free and PyObject_Realloc.
valgrind:
	valgrind --tool=memcheck --suppressions=./valgrind-python.supp \
   python3 -E -tt ./tests/test_matrix.py

depend:
	rm -f .depend.mk
	for f in $(CCFILES); do \
           $(CXX) -M -MT "$$(dirname $$f)/$$(basename -s .cc $$f).o" \
           $(CXXFLAGS) $$f >> .depend.mk; \
      done

-include .depend.mk
