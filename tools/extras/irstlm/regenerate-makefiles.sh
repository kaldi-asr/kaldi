#!/bin/bash

# NOTE:
# Versions 1.9 (or higher) of aclocal and automake are required.
# Version 2.59 (or higher) of autoconf is required.

# For Mac OSX users:
# Standard distribution usually includes versions 1.6 for aclocal and automake.
# Get versions 1.9 or higher
# Set the following variable to the correct paths
#ACLOCAL="/path/to/aclocal-1.9"
#AUTOMAKE="/path/to/automake-1.9"

force=$1;
# set parameter force to the value "--force" if you want to recreate all links to the autotools

die () {
  echo "$@" >&2
  exit 1
}

if [ -z "$ACLOCAL" ]
then
    ACLOCAL=`which aclocal`
fi

if [ -z "$AUTOMAKE" ]
then
    AUTOMAKE=`which automake`
fi

if [ -z "$AUTORECONF" ]
then
    AUTORECONF=`which autoreconf`
fi

if [ -z "$AUTOCONF" ]
then
    AUTOCONF=`which autoconf`
fi

if [ -z "$LIBTOOLIZE" ]
then
    LIBTOOLIZE=`which libtoolize`

    if [ -z "$LIBTOOLIZE" ]
    then
        LIBTOOLIZE=`which glibtoolize`
    fi
fi

if [ ! -d m4 ] ;
then
mkdir m4
fi

echo "Calling $AUTORECONF"
$AUTORECONF

ret=$?

if [ $ret -ne 0 ] ; then
echo "autoreconf FAILED"
echo "trying '$LIBTOOLIZE --force; $AUTOMAKE --add-missing ; $AUTORECONF'"
$LIBTOOLIZE --force
$AUTOMAKE  --add-missing
$AUTORECONF
if [ ! -e config.guess ] ; then 
$AUTOMAKE  --add-missing
$AUTORECONF
fi
fi

#echo "Calling $LIBTOOLIZE $force"
#$LIBTOOLIZE $force || die "libtoolize failed"

#echo "Calling $ACLOCAL..."
#$ACLOCAL -I m4 || die "aclocal failed"

#echo "Calling $AUTOCONF..."
#$AUTOCONF || die "autoconf failed"

#echo "Calling $AUTOMAKE --add-missing..."
#$AUTOMAKE --add-missing || die "automake failed"
