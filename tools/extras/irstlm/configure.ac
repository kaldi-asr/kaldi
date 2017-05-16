AC_INIT([irstlm], [5.80.06])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

m4_pattern_allow([AM_PROG_AR],[AM_PROG_AR])
AM_PROG_AR

AC_ARG_ENABLE([doc],
   [AC_HELP_STRING([--enable-doc|--disable-doc], [Enable or Disable (default) creation of documentation])])

AC_ARG_ENABLE([trace],
   [AC_HELP_STRING([--enable-trace|--disable-trace], [Enable (default) or Disable trace info at run-time])])

AC_ARG_ENABLE([assert],
   [AC_HELP_STRING([--enable-assert|--disable-assert], [Enable (default) or Disable assert calls at run-time])])

AC_ARG_ENABLE([debugging],
   [AC_HELP_STRING([--enable-debugging|--disable-debugging], [Enable or Disable (default) debugging info ("-g -O2")])])

AC_ARG_ENABLE(profiling,
   [AC_HELP_STRING([--enable-profiling|--disable-profiling], [Enable or Disable (default) profiling info ("-pg")])])

AC_ARG_ENABLE(caching,
   [AC_HELP_STRING([--enable-caching|--disable-caching], [Enable or Disable (default) internal caches to store probs and other info])])

AC_ARG_ENABLE(output,
   [AC_HELP_STRING([--enable-output|--disable-output], [Enable (default) or Disable part of the output])])

AC_ARG_ENABLE(interpolatedsearch,
   [AC_HELP_STRING([--enable-interpolatedsearch|--disable-interpolatedsearch], [Enable or Disable (default) interpolated search for n-grams])])

AC_ARG_ENABLE(optimization,
   [AC_HELP_STRING([--enable-optimization|--disable-optimization], [Enable or Disable (default) optimization info ("-O3")])])

AC_ARG_ENABLE(cxx0,
   [AC_HELP_STRING([--enable-cxx0|--disable-cxx0], [Enable or Disable (default) c++0x dialect ("-std=c++0x")])])

AC_ARG_WITH(zlib,
            [AC_HELP_STRING([--with-zlib=PATH], [(optional) path to zlib])],
            [with_zlib=$withval],
            [with_zlib=no]
           )

AC_ARG_WITH(tracelevel,
            [AC_HELP_STRING([--with-tracelevel=VAL], [(optional) level of tracing; default 0, tracing disabled])],
            [with_tracelevel=$withval],
            [with_tracelevel=0]
           )

AM_CONDITIONAL([DOC_COMPILATION],false)
if test "x$enable_doc" = 'xyes'
then
    AC_CHECK_TOOL(PDFLATEX,pdflatex,"no")
    AC_CHECK_TOOL(BIBTEX,bibtex,"no")
    if test "x$PDFLATEX" != "xno" && test "x$BIBTEX" != "xno"
    then
        AM_CONDITIONAL([DOC_COMPILATION],true)
        AC_MSG_NOTICE([pdflatex and bibtex are available])
        AC_MSG_NOTICE([documentation will be created])
    else
        AC_MSG_NOTICE([either pdflatex or bibtex is not available])
        AC_MSG_NOTICE([documentation will not be created (default); get it through the website $WEBSITE])
    fi
else
    AC_MSG_NOTICE([documentation will not be created (default); get it through the website $WEBSITE])
fi


#### Use this if you want that the default is yes
#### if test "x$enable_foo" != 'xno'
#### Use this if you want that the default is no
#### if test "x$enable_foo" = 'xyes'


if test "x$with_tracelevel" != 'x0'
then
if test "x$enable_trace" != 'xno'
then
    TRACELEVEL=${with_tracelevel}
    AC_MSG_NOTICE([trace enabled (default); trace level is $TRACELEVEL])
else
    TRACELEVEL=0
    AC_MSG_NOTICE([trace disabled; trace level is overwritten to $TRACELEVEL; most regression tests will fail])
fi
else
if test "x$enable_trace" != 'xno'
then
    TRACELEVEL=1
    AC_MSG_NOTICE([trace enabled (default); trace level is overwritten to default value $TRACELEVEL])
else
    TRACELEVEL=0
    AC_MSG_NOTICE([trace disabled; trace level is $TRACELEVEL; most regression tests will fail])
fi
fi
CPPFLAGS="$CPPFLAGS -DTRACE_LEVEL=$TRACELEVEL"

if test "x$enable_assert" != 'xno'
then
    AC_MSG_NOTICE([assert enabled (default)])
    CPPFLAGS="$CPPFLAGS -DMY_ASSERT_FLAG" 
else
    AC_MSG_NOTICE([assert disabled])
    CPPFLAGS="$CPPFLAGS -UMY_ASSERT_FLAG" 
fi

if test "x$enable_debugging" = 'xyes'
then
    AC_MSG_NOTICE([generation of debugging symbols enabled, compilation with "-g -O2"])
    CPPFLAGS="$CPPFLAGS -g -O2"
else
    AC_MSG_NOTICE([generation of debugging symbols disabled (default), compilation without "-g", only "-O2"])
fi

if test "x$enable_profiling" = 'xyes'
then
    AC_MSG_NOTICE([profiling enabled, compilation with "-pg"])
    CPPFLAGS="$CPPFLAGS -pg"
    LDFLAGS="$LDFLAGS -pg"
else
    AC_MSG_NOTICE([profiling disabled (default)])
fi

if test "x$enable_caching" = 'xyes'
then
    AC_MSG_NOTICE([caching enabled])
    CPPFLAGS="$CPPFLAGS -DPS_CACHE_ENABLE -DLMT_CACHE_ENABLE -DMDIADAPTLM_CACHE_ENABLE";
    LDFLAGS="$LDFLAGS"
else
    AC_MSG_NOTICE([caching disabled (default)])
fi

if test "x$enable_output" = 'xno'
then
    AC_MSG_NOTICE([part of the output on stdout is suppressed])
    CPPFLAGS="$CPPFLAGS -DOUTPUT_SUPPRESSED";
    LDFLAGS="$LDFLAGS"
else
    AC_MSG_NOTICE([caching disabled (default)])
fi

if test "x$enable_interpolatedsearch" = 'xyes'
then
    AC_MSG_NOTICE([interpolated search enabled])
    CPPFLAGS="$CPPFLAGS -DINTERP_SEARCH";
    LDFLAGS="$LDFLAGS"
else
    AC_MSG_NOTICE([interpolated search disabled (default)])
fi

if test "x$enable_optimization" = 'xyes'
then
    AC_MSG_NOTICE([optimization enabled, compilation with "-O3"])
    CPPFLAGS="$CPPFLAGS -O3";
    LDFLAGS="$LDFLAGS -O3"
else
    AC_MSG_NOTICE([optimization disabled (default)])
fi

if test "x$enable_cxx0" != 'xno'
then
    AC_MSG_NOTICE([c++x0 dialect is enabled (default), compilation with "-DHAVE_CXX0 -std=c++0x "])
    CPPFLAGS="$CPPFLAGS -DHAVE_CXX0 -std=c++0x";
else
    AC_MSG_NOTICE([c++x0 dialect is disabled, compilation without "-std=c++0x" and with "-UHAVE_CXX0"])
    CPPFLAGS="$CPPFLAGS -UHAVE_CXX0";
fi

if test "x$with_zlib" != 'xno'
then
  CPPFLAGS="$CPPFLAGS -I${with_zlib}/include"
  LDFLAGS="$LDFLAGS -L${with_zlib}/lib"
fi

LIBS="$LIBS -lz"

AC_CONFIG_FILES([
Makefile
src/Makefile
scripts/Makefile
doc/Makefile
])

AC_SUBST(transform,'s/_lm/-lm/')

AC_OUTPUT()

AC_MSG_NOTICE([The software will be installed into $prefix])
