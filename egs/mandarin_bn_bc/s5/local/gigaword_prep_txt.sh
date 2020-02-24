#!/usr/bin/env bash

# Copyright 2019 Johns Hopkins Univeersity (author: Jinyi Yang)
# Apache 2.0

if [ $# != 2 ]; then
  echo "$0 <gigaword-dir> <giga-lang-dir>"
  exit 0;
fi

giga_dir=$1
giga_lang_dir=$2

[ ! -d $giga_lang_dir ] && mkdir -p $giga_lang_dir;

find $giga_dir -name "*.gz" > $giga_lang_dir/giga_trans.flist || exit "Faile to find files"

if [ `wc -l $giga_lang_dir/giga_trans.flist | cut -d " " -f1` == 0 ]; then
  echo "Empty file list : $giga_lang_dir/giga_trans.flist"
  exit 1;
fi

for f in `cat $giga_lang_dir/giga_trans.flist`
do
  fname=$(basename "$f" ".gz")
  gunzip -c $f | \
    python3 local/gigaword_text_parse.py > $giga_lang_dir/$fname.tmp.txt
done

cat $giga_lang_dir/*.tmp.txt > $giga_lang_dir/raw.text
rm $giga_lang_dir/*.tmp.txt

pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
  CC=gcc CXX=g++ python setup.py build
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi
cat $giga_lang_dir/raw.text |\
  perl local/mandarin_text_normalize.pl |\
  python local/mandarin_segment.py > $giga_lang_dir/filtered.text
cat $giga_lang_dir/filtered.text |\
  python local/mandarin_segment.py > $giga_lang_dir/segmented.text
mv $giga_lang_dir/segmented.text $giga_lang_dir/text
