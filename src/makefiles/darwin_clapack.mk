# Darwin (macOS) configuration

ifndef DEBUG_LEVEL
$(error DEBUG_LEVEL not defined.)
endif
ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef OPENFSTINC
$(error OPENFSTINC not defined.)
endif
ifndef OPENFSTLIBS
$(error OPENFSTLIBS not defined.)
endif

CLAPACKLIBS = $(CLAPACKROOT)/CLAPACK-3.2.1/lapack.a $(CLAPACKROOT)/CLAPACK-3.2.1/libcblaswr.a \
	      $(CLAPACKROOT)/CBLAS/lib/cblas.a \
	      $(CLAPACKROOT)/f2c_BLAS-3.8.0/blas.a $(CLAPACKROOT)/libf2c/libf2c.a

CXXFLAGS = -std=c++14 -I.. -I$(OPENFSTINC) -O1 $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK -I../../tools/CLAPACK \
           -msse -msse2 \
           -g

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

ifeq ($(DEBUG_LEVEL), 0)
CXXFLAGS += -DNDEBUG
endif
ifeq ($(DEBUG_LEVEL), 2)
CXXFLAGS += -O0 -DKALDI_PARANOID
endif

# Compiler specific flags
COMPILER = $(shell $(CXX) -v 2>&1)
ifeq ($(findstring clang,$(COMPILER)),clang)
# Suppress annoying clang warnings that are perfectly valid per spec.
CXXFLAGS += -Wno-mismatched-tags
else ifeq ($(findstring GCC,$(COMPILER)),GCC)
# Allow implicit conversions between vectors.
CXXFLAGS += -flax-vector-conversions
endif

LDFLAGS = $(EXTRA_LDFLAGS) $(OPENFSTLDFLAGS) -g
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(CLAPACKLIBS) -lm -ldl

ifneq ($(ARCH), WASM)
    CXXFLAGS += -pthread
    LDLIBS += -lpthread
endif
