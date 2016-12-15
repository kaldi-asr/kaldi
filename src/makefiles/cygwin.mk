# Cygwin settings

CXXFLAGS += -msse -msse2 -DHAVE_CLAPACK -I ../../tools/CLAPACK/

LDFLAGS += -g --enable-auto-import -L/usr/lib/lapack
LDLIBS += -lcyglapack-0 -lcygblas-0
