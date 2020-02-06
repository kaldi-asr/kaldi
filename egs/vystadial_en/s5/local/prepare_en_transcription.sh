#!/usr/bin/env bash

locdata=$1
locdict=$2

cmu_dict=common/cmudict.0.7a
cmu_ext=common/cmudict.ext

mkdir -p $locdict

if [ ! -f $cmu_dict ] ; then
  echo "--- Downloading CMU dictionary ..."
  svn export http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a \
     $cmu_dict || exit 1;
fi

echo; echo "If common/cmudict.ext exists, add extra pronunciation to dictionary" ; echo
cat $cmu_dict  $cmu_ext > $locdict/cmudict_ext.txt 2> /dev/null  # ignoring if no extension

echo "--- Striping stress and pronunciation variant markers from cmudict ..."
perl local/make_baseform.pl \
  $locdict/cmudict_ext.txt /dev/stdout |\
  sed -e 's:^\([^\s(]\+\)([0-9]\+)\(\s\+\)\(.*\):\1\2\3:' > $locdict/cmudict-plain.txt

echo "--- Searching for OOV words ..."
gawk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $locdict/cmudict-plain.txt $locdata/vocab-full.txt |\
  egrep -v '<.?s>' > $locdict/vocab-oov.txt

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $locdata/vocab-full.txt $locdict/cmudict-plain.txt |\
  egrep -v '<.?s>' > $locdict/lexicon.txt

wc -l $locdict/vocab-oov.txt
wc -l $locdict/lexicon.txt
