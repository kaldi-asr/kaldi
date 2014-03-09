# makefiles/darwin_10_9.mk contains Darwin-specific rules for OS X 10.9.*

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

CXXFLAGS = -msse -msse2 -Wall -I.. \
	  -fPIC \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
      -DHAVE_CLAPACK \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID

LDFLAGS = -g
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -lm -lpthread -framework Accelerate
CXX = g++
CC = $(CXX)
RANLIB = ranlib
AR = ar

# On Mac OS 10.9, g++ is actually clang in disguise which by default uses the
# new c++ standard library libc++. Since openfst uses stuff from the tr1
# namespace, we need to tell clang to use libstdc++ instead.
COMPILER = $(shell $(CXX) -v 2>&1 )
ifeq ($(findstring clang,$(COMPILER)),clang)
	CXXFLAGS += -stdlib=libstdc++
	LDFLAGS += -stdlib=libstdc++
endif

# We need to tell recent versions of g++ to allow vector conversions without 
# an explicit cast provided the vectors are of the same size.
ifeq ($(findstring GCC,$(COMPILER)),GCC)
	CXXFLAGS += -flax-vector-conversions -Wno-unused-local-typedefs
endif
