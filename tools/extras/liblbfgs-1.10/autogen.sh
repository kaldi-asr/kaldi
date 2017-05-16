#!/bin/sh
# $Id$

if [ "$1" = "--force" ];
then
    FORCE=--force
    NOFORCE=
    FORCE_MISSING=--force-missing
else
    FORCE=
    NOFORCE=--no-force
    FORCE_MISSING=
fi

libtoolize --copy $FORCE 2>&1 | sed '/^You should/d' || {
    echo "libtoolize failed!"
    exit 1
}

aclocal $FORCE || {
    echo "aclocal failed!"
    exit 1
}

autoheader $FORCE || {
    echo "autoheader failed!"
    exit 1
}

automake -a -c $NOFORCE || {
    echo "automake failed!"
    exit 1
}

autoconf $FORCE || {
    echo "autoconf failed!"
    exit 1
}
