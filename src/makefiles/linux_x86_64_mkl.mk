# MKL specific Linux configuration

# The linking flags for MKL will be very different depending on the OS,
# architecture, compiler, etc. used. In case configure did not cut it, use
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/

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

CXXFLAGS = -std=c++17 -I.. -isystem $(OPENFSTINC) -O1 \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_MKL $(MKL_CXXFLAGS) \
           -m64 -msse -msse2 -pthread -g

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

# As late as possible to allow the user to do what they want.
CXXFLAGS += $(EXTRA_CXXFLAGS)

LDFLAGS = $(OPENFSTLDFLAGS) -rdynamic $(EXTRA_LDFLAGS)
LDLIBS =  $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(MKL_LDLIBS) -lm -lpthread -ldl
