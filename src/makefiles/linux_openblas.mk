# OpenBLAS specific Linux settings

ifndef OPENBLASLIBS
$(error OPENBLASLIBS not defined.)
endif

ifndef OPENBLASROOT
$(error OPENBLASROOT not defined.)
endif

CXXFLAGS += -msse -msse2 -pthread -rdynamic \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
            -DHAVE_OPENBLAS -I $(OPENBLASROOT)/include

LDFLAGS += -rdynamic
LDLIBS += $(OPENBLASLIBS)
