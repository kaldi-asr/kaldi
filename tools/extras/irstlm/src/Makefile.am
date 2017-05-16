lib_LTLIBRARIES = libirstlm.la

AM_CXXFLAGS = -static -isystem/usr/include -W -Wall -ffor-scope -D_FILE_OFFSET_BITS=64 -D_LARGE_FILES $(BOOST_CPPFLAGS) -DMYCODESIZE=3

libirstlm_ladir = ${includedir}

libirstlm_la_HEADERS = \
  cmd.h \
  thpool.h \
  dictionary.h \
  gzfilebuf.h \
  htable.h \
  index.h \
  lmContainer.h \
  lmclass.h \
  lmmacro.h \
  lmtable.h \
  lmInterpolation.h \
  mempool.h \
  mfstream.h \
  n_gram.h \
  ngramcache.h \
  ngramtable.h \
  timer.h \
  util.h \
  crc.h \
  interplm.h \
  linearlm.h \
  mdiadapt.h \
  mixture.h \
  normcache.h \
  shiftlm.h \
  cplsa.h \
  cswam.h \
  doc.h

libirstlm_la_SOURCES = \
  cmd.c \
  thpool.c \
  dictionary.cpp \
  htable.cpp \
  lmContainer.cpp \
  lmclass.cpp \
  lmmacro.cpp \
  lmtable.cpp \
  lmInterpolation.cpp \
  mempool.cpp \
  mfstream.cpp \
  n_gram.cpp \
  ngramcache.cpp \
  ngramtable.cpp \
  timer.cpp \
  util.cpp \
  crc.cpp \
  interplm.cpp \
  linearlm.cpp \
  mdiadapt.cpp \
  mixture.cpp \
  normcache.cpp \
  shiftlm.cpp \
  cplsa.cpp \
  cswam.cpp \
  doc.cpp

CLEANFILES = $(BUILT_SOURCES)

libirstlm_la_LIBADD = $(BOOST_LDFLAGS) $(BOOST_THREAD_LIB)

LDADD = -lirstlm -lpthread
DEPENDENCIES = libirstlm.la

bin_PROGRAMS = dict ngt dtsel compile-lm interpolate-lm prune-lm quantize-lm prune-lm score-lm tlm plsa verify-caching cswa
dict_SOURCES = dict.cpp
dict_DEPENDENCIES = $(DEPENDENCIES)
ngt_SOURCES = ngt.cpp
ngt_DEPENDENCIES = $(DEPENDENCIES)
dtsel_SOURCES = dtsel.cpp
dtsel_DEPENDENCIES = $(DEPENDENCIES)
compile_lm_SOURCES = compile-lm.cpp
compile_lm_DEPENDENCIES = $(DEPENDENCIES)
interpolate_lm_SOURCES = interpolate-lm.cpp
interpolate_lm_DEPENDENCIES = $(DEPENDENCIES)
prune_lm_SOURCES = prune-lm.cpp
prune_lm_DEPENDENCIES = $(DEPENDENCIES)
quantize_lm_SOURCES = quantize-lm.cpp
quantize_lm_DEPENDENCIES = $(DEPENDENCIES)
score_lm_SOURCES = score-lm.cpp
score_lm_DEPENDENCIES = $(DEPENDENCIES)
tlm_SOURCES = tlm.cpp
tlm_DEPENDENCIES = $(DEPENDENCIES)
plsa_SOURCES = plsa.cpp
plsa_DEPENDENCIES = $(DEPENDENCIES)
verify_caching_SOURCES = verify-caching.cpp
verify_caching_DEPENDENCIES = $(DEPENDENCIES)
cswa_SOURCES = cswa.cpp
cswa_DEPENDENCIES = $(DEPENDENCIES)
