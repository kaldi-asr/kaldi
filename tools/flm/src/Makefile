#
#    File:   Makefile.example
#    Author: The SRI DECIPHER (TM) System
#    Date:   Thu Sep  9 12:04:47 1993
#
#    Description:
#	This is the example makefile to start from when adding new
#       modules to the DECIPHER System.  To use this makefile, first
#       copy it to your directory as the file "Makefile".  Second,
#       replace the word "Example" in the text below with the real name
#	of your library.  Next replace the the example filenames with
#	the names of your actual declarations and source files in the
#	appropriate variable definitions.  Finally clean up by deleting
#	any lines not relevant to your module and updating this header
#       to describe your new module.  Do not forget to use the proper
#       RCS keywords!
#
#    Copyright (c) 1993, SRI International.  All Rights Reserved.
#
#    $Header: /home/srilm/CVS/srilm/flm/src/Makefile,v 1.15 2013/03/31 06:00:37 stolcke Exp $
#

# Include common SRILM variable definitions.
include $(SRILM)/common/Makefile.common.variables

# This should enable locale-specific string collation for vocabulary sorting
# (it will slow things down somewhat).
#ADDITIONAL_CXXFLAGS = -Dstrcmp=strcoll

# Flags for generating "compact" data structures
COMPACT_FLAGS += -DUSE_SARRAY -DUSE_SARRAY_TRIE -DUSE_SARRAY_MAP2

# Flags for generating "short" data structures
SHORT_FLAGS = $(COMPACT_FLAGS) -DUSE_SHORT_VOCAB -DUSE_XCOUNTS

# Flags for generating "long long" data structures
LLONG_FLAGS = $(COMPACT_FLAGS) -DUSE_LONGLONG_COUNTS -DUSE_XCOUNTS

# Define variables.

TEMPLATE_SOURCES = 

# Example Library declarations files.

EXTERNAL_LIB_HEADERS = \
	$(SRCDIR)/hexdec.h \
	$(SRCDIR)/FDiscount.h \
	$(SRCDIR)/FNgramStats.h \
	$(SRCDIR)/FNgramSpecs.h \
	$(SRCDIR)/FactoredVocab.h \
	$(SRCDIR)/FNgram.h \
	$(SRCDIR)/ProductVocab.h \
	$(SRCDIR)/ProductNgram.h \
	$(SRCDIR)/FLMThreads.h \
	$(SRCDIR)/wmatrix.h

# Example Library source files.
LIB_SOURCES = 				\
	$(SRCDIR)/FDiscount.cc \
	$(SRCDIR)/FNgramStats.cc \
	$(SRCDIR)/FNgramStatsInt.cc \
	$(SRCDIR)/FNgramSpecs.cc \
	$(SRCDIR)/FNgramSpecsInt.cc \
	$(SRCDIR)/FactoredVocab.cc \
	$(SRCDIR)/FNgramLM.cc \
	$(SRCDIR)/ProductVocab.cc \
	$(SRCDIR)/ProductNgram.cc \
	$(SRCDIR)/FLMThreads.cc \
	$(SRCDIR)/strtolplusb.cc \
	$(SRCDIR)/wmatrix.cc

# Example Library object files.
LIB_OBJECTS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%$(OBJ_SUFFIX),\
		$(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%$(OBJ_SUFFIX),$(LIB_SOURCES)))

# Example Library.
LIBRARY = $(OBJDIR)/$(LIB_PREFIX)flm$(LIB_SUFFIX)

ADDITIONAL_INCLUDES += \
	$(TCL_INCLUDE)

ADDITIONAL_LDFLAGS += \
	$(MATHERR_LINK)

ADDITIONAL_LIBRARIES += \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)oolm$(LIB_SUFFIX) \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)dstruct$(LIB_SUFFIX) \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)misc$(LIB_SUFFIX) \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)z$(LIB_SUFFIX) \
	$(TCL_LIBRARY) \
	$(MATH_LIBRARY) \
	$(LBFGS_LIBRARY)

# Example programs.

# compile a version of ngram here for convenient testing only
TEST_PROGRAM_NAMES = \
	pngram

REAL_PROGRAM_NAMES = \
	fngram-count \
	fngram 

PROGRAM_NAMES = \
	$(TEST_PROGRAM_NAMES) \
	$(REAL_PROGRAM_NAMES)

PROGRAMS = $(PROGRAM_NAMES:%=$(BINDIR)/%$(EXE_SUFFIX))

PROGRAM_SOURCES = $(foreach prog,$(PROGRAM_NAMES),\
			$(wildcard $(SRCDIR)/$(prog).c) \
			$(wildcard $(SRCDIR)/$(prog).cc))
PROGRAM_OBJECTS = $(PROGRAM_NAMES:%=$(OBJDIR)/%$(OBJ_SUFFIX))


# Libraries to be linked with the Example programs.
LIBRARIES = $(LIBRARY) \
	    $(ADDITIONAL_LIBRARIES)

# All of the types of files.

ALL_HEADERS = $(EXTERNAL_LIB_HEADERS) \
	      $(INTERNAL_LIB_HEADERS)

ALL_SOURCES = $(LIB_SOURCES) \
	      $(PROGRAM_SOURCES)

ALL_OBJECTS = $(LIB_OBJECTS) \
	      $(PROGRAM_OBJECTS)

ALL_LIBRARIES = $(LIBRARY)

ALL_PROGRAMS = $(PROGRAMS)

ALL_PROGRAM_NAMES = $(PROGRAM_NAMES)


# Define pseudo-targets.


# Make sure the library does not get deleted if the make is interrupted.
.PRECIOUS:	$(LIBRARY)


# Define targets.


all:	$(PROGRAMS)

$(LIBRARY):	$(LIB_OBJECTS)
	$(ARCHIVE) $(AR_OUTPUT_OPTION) $^ $(DEMANGLE_FILTER)
	$(RANLIB) $@ $(DEMANGLE_FILTER)

$(PROGRAMS): $(LIBRARY) $(OTHER_LIBRARIES)

# Variables and Targets for released system

EXPORTED_HEADERS   = $(EXTERNAL_LIB_HEADERS)
EXPORTED_LIBRARIES = $(LIBRARY)
EXPORTED_PROGRAMS  = $(REAL_PROGRAM_NAMES:%=$(BINDIR)/%$(EXE_SUFFIX))

release:	release-headers release-libraries release-programs

# Include common SRILM target definitions.
include $(SRILM)/common/Makefile.common.targets
