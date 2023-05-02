# Cygwin configuration

ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef OPENFSTINC
$(error OPENFSTINC not defined.)
endif
ifndef OPENFSTLIBS
$(error OPENFSTLIBS not defined.)
endif

CXXFLAGS = -std=c++14 -U__STRICT_ANSI__ -I.. -I$(OPENFSTINC) -O1 $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_CLAPACK -I../../tools/CLAPACK/ \
           -msse -msse2 -O -Wa,-mbig-obj \
           -g # -O0 -DKALDI_PARANOID

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = $(EXTRA_LDFLAGS) $(OPENFSTLDFLAGS) -g \
          --enable-auto-import -L/usr/lib/lapack
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) -lcyglapack-0 -lcygblas-0 \
         -lm -lpthread -ldl
