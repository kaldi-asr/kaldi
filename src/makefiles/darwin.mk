# Darwin (macOS) settings

CXXFLAGS += -msse -msse2 -pthread \
            -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK

# Compiler specific flags
COMPILER = $(shell $(CXX) -v 2>&1)
ifeq ($(findstring clang,$(COMPILER)),clang)
# Suppress annoying clang warnings that are perfectly valid per spec.
CXXFLAGS += -Wno-mismatched-tags
else ifeq ($(findstring GCC,$(COMPILER)),GCC)
# Allow implicit conversions between vectors.
CXXFLAGS += -flax-vector-conversions
endif

LDFLAGS += -g
LDLIBS += -framework Accelerate
