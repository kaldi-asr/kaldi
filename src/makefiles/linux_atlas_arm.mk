# ATLAS specific Linux ARM settings

ifndef ATLASINC
$(error ATLASINC not defined.)
endif

ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif

CXXFLAGS += -ftree-vectorize -mfloat-abi=hard -mfpu=neon -pthread -rdynamic \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$(ATLASINC)

LDFLAGS += -rdynamic
LDLIBS += $(ATLASLIBS)
