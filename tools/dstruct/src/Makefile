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
#    $Header: /home/srilm/CVS/srilm/dstruct/src/Makefile,v 1.41 2014-05-27 03:05:34 stolcke Exp $
#

# Include common SRILM variable definitions.
include $(SRILM)/common/Makefile.common.variables

# Flags for generating "compact" data structures
COMPACT_FLAGS += -DUSE_SARRAY -DUSE_SARRAY_TRIE -DUSE_SARRAY_MAP2

# Flags for generating "short" data structures
SHORT_FLAGS = $(COMPACT_FLAGS) -DUSE_SHORT_VOCAB -DUSE_XCOUNTS

# Flags for generating "long long" data structures
LLONG_FLAGS = $(COMPACT_FLAGS) -DUSE_LONGLONG_COUNTS -DUSE_XCOUNTS

# Define variables.

TEMPLATE_SOURCES =	\
	$(SRCDIR)/Array.cc \
	$(SRCDIR)/IntervalHeap.cc \
	$(SRCDIR)/Map.cc \
	$(SRCDIR)/SArray.cc \
	$(SRCDIR)/LHash.cc \
	$(SRCDIR)/Map2.cc \
	$(SRCDIR)/Trie.cc \
	$(SRCDIR)/CachedMem.cc

# Example Library declarations files.

EXTERNAL_LIB_HEADERS = \
	$(SRCDIR)/Array.h		\
	$(SRCDIR)/IntervalHeap.h	\
	$(SRCDIR)/MemStats.h		\
	$(SRCDIR)/Map.h			\
	$(SRCDIR)/SArray.h		\
	$(SRCDIR)/LHash.h		\
	$(SRCDIR)/Map2.h		\
	$(SRCDIR)/Trie.h		\
	$(SRCDIR)/CachedMem.h		\
	$(SRCDIR)/BlockMalloc.h		\
	$(SRCDIR)/DStructThreads.h	\
	$(TEMPLATE_SOURCES)

INTERNAL_LIB_HEADERS = 

# Example Library source files.
LIB_SOURCES = 				\
	$(SRCDIR)/qsort.c \
	$(SRCDIR)/MemStats.cc \
	$(SRCDIR)/LHashTrie.cc \
	$(SRCDIR)/SArrayTrie.cc \
	$(SRCDIR)/BlockMalloc.cc \
	$(SRCDIR)/DStructThreads.cc \
	$(TEMPLATE_SOURCES)

# Example Library object files.
LIB_OBJECTS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%$(OBJ_SUFFIX),\
		$(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%$(OBJ_SUFFIX),$(LIB_SOURCES)))


# Example Library.
LIBRARY = $(OBJDIR)/$(LIB_PREFIX)dstruct$(LIB_SUFFIX)

ADDITIONAL_INCLUDES += \
	$(TCL_INCLUDE)

ADDITIONAL_LIBRARIES += \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)misc$(LIB_SUFFIX) \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)z$(LIB_SUFFIX) \
	$(TCL_LIBRARY) \
	$(MATH_LIBRARY)

# Exported programs.
REAL_PROGRAM_NAMES = \
		maxalloc

# Example programs.
PROGRAM_NAMES = \
		$(REAL_PROGRAM_NAMES) \
		testArray \
		testMap \
		testFloatMap \
		benchHash \
		testHash \
		testSizes \
		testCachedMem \
		testBlockMalloc

ifeq ($(NO_TCL), )
PROGRAM_NAMES += \
		testMap2 \
		testTrie
endif

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
	$(ARCHIVE) $(AR_OUTPUT_OPTION) $^
	$(RANLIB) $@ $(DEMANGLE_FILTER)

$(PROGRAMS):	$(LIBRARY) $(OTHER_LIBRARIES)

# Variables and Targets for released system

EXPORTED_HEADERS   = $(EXTERNAL_LIB_HEADERS)
EXPORTED_LIBRARIES = $(LIBRARY)
EXPORTED_PROGRAMS  = $(REAL_PROGRAM_NAMES:%=$(BINDIR)/%$(EXE_SUFFIX))

release:	release-headers release-libraries release-programs

# Include common SRILM target definitions.
include $(SRILM)/common/Makefile.common.targets
