#!/bin/bash

cmudir=data/local/cmudict
cmuurl=http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/
cmuver=cmudict.0.7a

mkdir -p $cmudir

if [ ! -f $cmudir/$cmuver ]; then
  wget -O $cmudir/$cmuver svn $cmuurl/$cmuver
  wget -O $cmudir/$cmuver.phones svn $cmuurl/$cmuver.phones
  wget -O $cmudir/$cmuver.symbols svn $cmuurl/$cmuver.symbols
fi

#remove comments
grep -e "^;;;" -v $cmudir/$cmuver > $cmudir/cmudict.full

#limit dictionary

