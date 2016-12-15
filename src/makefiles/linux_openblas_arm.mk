# OpenBLAS specific Linux ARM settings

ifndef OPENBLASLIBS
$(error OPENBLASLIBS not defined.)
endif

ifndef OPENBLASROOT
$(error OPENBLASROOT not defined.)
endif

CXXFLAGS += -ftree-vectorize -mfloat-abi=hard -mfpu=neon -pthread -rdynamic \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
            -DHAVE_OPENBLAS -I $(OPENBLASROOT)/include

LDFLAGS += -rdynamic
LDLIBS += $(OPENBLASLIBS)
