dnl $Id$
dnl
dnl
dnl Exported and configured variables:
dnl CFLAGS
dnl LDFLAGS
dnl INCLUDES


dnl ------------------------------------------------------------------
dnl Initialization for autoconf
dnl ------------------------------------------------------------------
AC_PREREQ(2.59)
AC_INIT
AC_CONFIG_SRCDIR([lib/lbfgs.c])


dnl ------------------------------------------------------------------
dnl Initialization for automake
dnl ------------------------------------------------------------------
AM_INIT_AUTOMAKE(liblbfgs, 1.10)
AC_CONFIG_HEADERS(config.h)
AM_MAINTAINER_MODE


dnl ------------------------------------------------------------------
dnl Checks for program
dnl ------------------------------------------------------------------
AC_PROG_LIBTOOL
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET


dnl ------------------------------------------------------------------
dnl Initialization for variables
dnl ------------------------------------------------------------------
CFLAGS="${ac_save_CFLAGS} -Wall"
LDFLAGS="${ac_save_LDFLAGS}"
INCLUDES="-I\$(top_srcdir) -I\$(top_srcdir)/include"



dnl ------------------------------------------------------------------
dnl Checks for header files.
dnl ------------------------------------------------------------------
AC_HEADER_STDC
AC_CHECK_HEADERS(xmmintrin.h emmintrin.h)



dnl ------------------------------------------------------------------
dnl Checks for debugging mode
dnl ------------------------------------------------------------------
AC_ARG_ENABLE(
    debug,
    [AS_HELP_STRING(
        [--enable-debug],
        [build for debugging]
        )],
    [CFLAGS="-DDEBUG -O -g ${CFLAGS}"],
    [CFLAGS="-O3 -ffast-math ${CFLAGS}"]
)

dnl ------------------------------------------------------------------
dnl Checks for profiling mode
dnl ------------------------------------------------------------------
AC_ARG_ENABLE(
    profile,
    [AS_HELP_STRING(
        [--enable-profile],
        [build for profiling]
	)],
    [CFLAGS="-DPROFILE -pg ${CFLAGS}"]
)

dnl ------------------------------------------------------------------
dnl Checks for SSE2 build
dnl ------------------------------------------------------------------
AC_ARG_ENABLE(
    sse2,
    [AS_HELP_STRING(
        [--enable-sse2],
        [enable SSE2 optimization routines]
        )],
    [CFLAGS="-msse2 -DUSE_SSE ${CFLAGS}"]
)

dnl ------------------------------------------------------------------
dnl Checks for library functions.
dnl ------------------------------------------------------------------
AC_CHECK_LIB(m, fabs)



dnl ------------------------------------------------------------------
dnl Export variables
dnl ------------------------------------------------------------------
AC_SUBST(CFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(INCLUDES)

dnl ------------------------------------------------------------------
dnl Output the configure results.
dnl ------------------------------------------------------------------
AC_CONFIG_FILES(Makefile lib/Makefile sample/Makefile)
AC_OUTPUT
