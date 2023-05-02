# OpenBLAS specific Linux ARM configuration

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
ifndef OPENBLASINC
$(error OPENBLASINC not defined.)
endif
ifndef OPENBLASLIBS
$(error OPENBLASLIBS not defined.)
endif

CXXFLAGS = -std=c++14 -I.. -isystem $(OPENFSTINC) -O1 $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_OPENBLAS -I$(OPENBLASINC) \
           -ftree-vectorize -mfloat-abi=hard -mfpu=neon -pthread \
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
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(OPENBLASLIBS) -lm -lpthread -ldl
