# CLAPACK specific Linux ARM settings

CXXFLAGS += -ftree-vectorize -mfloat-abi=hard -mfpu=neon -pthread -rdynamic \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
            -DHAVE_CLAPACK -I ../../tools/CLAPACK

LDFLAGS += -rdynamic
LDLIBS += $(ATLASLIBS)
