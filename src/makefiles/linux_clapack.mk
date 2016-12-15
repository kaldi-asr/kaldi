# CLAPACK specific Linux settings

CXXFLAGS += -msse -msse2 -pthread -rdynamic \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H \
            -DHAVE_CLAPACK -I ../../tools/CLAPACK

LDFLAGS += -rdynamic
LDLIBS += $(ATLASLIBS)
