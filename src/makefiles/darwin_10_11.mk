# makefiles/darwin_10_11.mk contains Darwin-specific rules for OS X 10.11.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

DOUBLE_PRECISION = 0
CXXFLAGS += -msse -msse2 -Wall -I.. \
      -pthread \
      -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) -Wno-unused-local-typedef \
      -g # -O0 -DKALDI_PARANOID


ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -g
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
CXX = g++
CC = $(CXX)
RANLIB = ranlib
AR = ar

# Add no-mismatched-tags flag to suppress the annoying clang warnings
# that are perfectly valid per spec.
COMPILER = $(shell $(CXX) -v 2>&1 )
ifeq ($(findstring clang,$(COMPILER)),clang)
  CXXFLAGS += -Wno-mismatched-tags
  # Link with libstdc++ if we are building against OpenFst < 1.4
  ifneq ("$(OPENFST_GE_10400)","1")
    CXXFLAGS += -stdlib=libstdc++
    LDFLAGS += -stdlib=libstdc++
  endif
endif

# We need to tell recent versions of g++ to allow vector conversions without
# an explicit cast provided the vectors are of the same size.
ifeq ($(findstring GCC,$(COMPILER)),GCC)
	CXXFLAGS += -flax-vector-conversions -Wno-unused-local-typedefs
endif
