# Platform independent settings

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif

ifndef OPENFSTLIBS
$(error OPENFSTLIBS not defined.)
endif

CXXFLAGS = -std=c++11 -I.. -I$(FSTROOT)/include \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           $(EXTRA_CXXFLAGS) \
           -g # -O0 -DKALDI_PARANOID

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = $(OPENFSTLDFLAGS) $(EXTRA_LDFLAGS)
LDLIBS = $(OPENFSTLIBS) -lm -lpthread -ldl $(EXTRA_LDLIBS)

RANLIB = ranlib
AR = ar
AS = as
