# MKL specific Linux configuration

# We have tested Kaldi with MKL version 10.2 on Linux/GCC and Intel(R) 64
# architecture (also referred to as x86_64) with LP64 interface layer.

# The linking flags for MKL will be very different depending on the OS,
# architecture, compiler, etc. used. The correct flags can be obtained from
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
# Use the options obtained from this website to manually configure for other
# platforms using MKL.

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
ifndef MKLROOT
$(error MKLROOT not defined.)
endif

MKLLIB ?= $(MKLROOT)/lib/intel64

CXXFLAGS = -std=c++11 -I.. -isystem $(OPENFSTINC) -O1 $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_MKL -I$(MKLROOT)/include \
           -m64 -msse -msse2 -pthread \
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
endif

LDFLAGS = $(EXTRA_LDFLAGS) $(OPENFSTLDFLAGS) -rdynamic
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(MKLFLAGS) -lm -lpthread -ldl
