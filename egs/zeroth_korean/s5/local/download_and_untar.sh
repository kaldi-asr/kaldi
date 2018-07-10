#!/bin/bash

# Copyright 2018 Lucas Jo (Atlas Guide)
#           2018 Wonkyum Lee (Gridspace)
# Apache 2.0

if [ $# -ne "1" ]; then
	echo "Usage: $0 <download_dir>"
	echo "e.g.: $0 ./db"
	exit 1
fi

exists(){
	command -v "$1" >/dev/null 2>&1
}


dir=$1
local_lm_dir=data/local/lm

AUDIOINFO='AUDIO_INFO'
AUDIOLIST='train_data_01 test_data_01'

echo "Now download corpus ----------------------------------------------------"
if [ ! -f $dir/db.tar.gz ]; then
  if [ ! -d $dir ]; then 
    mkdir -p $dir
  fi
  wget -O $dir/db.tar.gz http://www.openslr.org/resources/40/zeroth_korean.tar.gz 
else
  echo "  $dir/db.tar.gz already exist"
fi

echo "Now extract corpus ----------------------------------------------------"
if [ ! -f $dir/$AUDIOINFO ]; then
  tar -zxvf $dir/db.tar.gz -C $dir
  else
    echo "  corpus already extracted"
fi

if [ ! -d $local_lm_dir ]; then
    mkdir -p $local_lm_dir
fi
echo "Check LMs files"
LMList="\
  zeroth.lm.fg.arpa.gz \
  zeroth.lm.tg.arpa.gz \
  zeroth.lm.tgmed.arpa.gz \
  zeroth.lm.tgsmall.arpa.gz \
  zeroth_lexicon \
  zeroth_morfessor.seg"

for file in $LMList; do
  if [ -f $local_lm_dir/$file ]; then
    echo $file already exist
  else
    echo "Linking "$file
    ln -s $PWD/$dir/$file $local_lm_dir/$file
  fi
done
echo "all the files (lexicon, LM, segment model) are ready"
