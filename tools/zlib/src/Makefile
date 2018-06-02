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
#    $Header: /home/srilm/CVS/srilm/zlib/src/Makefile,v 1.1 2013/03/19 20:05:46 stolcke Exp $
#

# Include common SRILM variable definitions.
include $(SRILM)/common/Makefile.common.variables

# Define variables.

# Example Library declarations files.

EXTERNAL_LIB_HEADERS = \
	$(SRCDIR)/zlib.h          	\
	$(SRCDIR)/zconf.h		

INTERNAL_LIB_HEADERS =  \
	$(SRCDIR)/crc32.h  		\
	$(SRCDIR)/deflate.h  		\
	$(SRCDIR)/gzguts.h  		\
	$(SRCDIR)/inffast.h  		\
	$(SRCDIR)/inffixed.h  		\
	$(SRCDIR)/inflate.h  		\
	$(SRCDIR)/inftrees.h  		\
	$(SRCDIR)/trees.h  		\
	$(SRCDIR)/zutil.h

# Example Library source files.
LIB_SOURCES = 				\
	$(SRCDIR)/adler32.c   		\
	$(SRCDIR)/crc32.c    		\
	$(SRCDIR)/gzclose.c  		\
	$(SRCDIR)/gzread.c   		\
	$(SRCDIR)/infback.c  		\
	$(SRCDIR)/inflate.c   		\
	$(SRCDIR)/trees.c    		\
	$(SRCDIR)/zutil.c			\
	$(SRCDIR)/compress.c  		\
	$(SRCDIR)/deflate.c  		\
	$(SRCDIR)/gzlib.c    		\
	$(SRCDIR)/gzwrite.c  		\
	$(SRCDIR)/inffast.c  		\
	$(SRCDIR)/inftrees.c  		\
	$(SRCDIR)/uncompr.c


# Example Library object files.
LIB_OBJECTS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%$(OBJ_SUFFIX),\
		$(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%$(OBJ_SUFFIX),$(LIB_SOURCES)))

# Example Library.
LIBRARY = $(OBJDIR)/$(LIB_PREFIX)z$(LIB_SUFFIX)

ADDITIONAL_INCLUDES += \
	$(TCL_INCLUDE)

ADDITIONAL_LIBRARIES += \
	$(TCL_LIBRARY) \
	$(MATH_LIBRARY)

# Example programs.
PROGRAM_NAMES = \
	minigzip

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


all:	libraries

$(LIBRARY):	$(LIB_OBJECTS)
	$(ARCHIVE) $(AR_OUTPUT_OPTION) $^
	$(RANLIB) $@ $(DEMANGLE_FILTER)

# Variables and Targets for released system

EXPORTED_HEADERS   = $(EXTERNAL_LIB_HEADERS)
EXPORTED_LIBRARIES = $(LIBRARY)
EXPORTED_PROGRAMS  = 

release:	release-headers release-libraries release-programs

# Include common SRILM target definitions.
include $(SRILM)/common/Makefile.common.targets

