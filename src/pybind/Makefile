
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
CXXFLAGS += -O3 -flto
LDFLAGS += -flto
# TODO:

ifeq ($(shell uname),Darwin)
  LDFLAGS += -undefined dynamic_lookup
endif

CCFILES = kaldi_pybind.cc \
          matrix/matrix_common_pybind.cc matrix/matrix_pybind.cc \
          matrix/vector_pybind.cc \
          util/table_types_pybind.cc

LIBNAME = kaldi_pybind


ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

EXTRA_LDLIBS += $(foreach dep,$(ADDLIBS), $(dir $(dep))lib$(notdir $(basename $(dep))).so)

LIBFILE=$(LIBNAME)$(LIBFILE_EXTENSION)

.PHONY: all clean test

all: $(LIBFILE)

$(LIBFILE): $(ADDLIBS) $(CCFILES)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(CCFILES) -Wl,--no-whole-archive -Wl,-rpath=$(CURDIR)/../lib $(LDFLAGS) $(LDLIBS) $(EXTRA_LDLIBS)
	python3 -c 'import kaldi_pybind'  # this line is a test.

clean:
	-rm -f *.so
	-rm -rf __pycache__

test: all
	./tests/test_kaldi_pybind.py
	./tests/test_matrix.py

# valgrind-python.supp is from http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp
# since we do not compile Python from souce, we follow the comment in valgrind-python.supp
# to uncomment suppressions for PyObject_Free and PyObject_Realloc.
valgrind:
	valgrind --tool=memcheck --suppressions=./valgrind-python.supp \
   python3 -E -tt ./tests/test_matrix.py
