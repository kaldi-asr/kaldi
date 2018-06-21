#!/bin/bash

tmpdir=data/local/tmp
download_dir=$tmpdir/subs
mkdir -p $download_dir
subs_src="http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en-es.txt.zip"

# download the subs corpus
if [ ! -f $download_dir/subs.zip ]; then
  wget -O $download_dir/subs.zip $subs_src
  (
    cd $download_dir
    unzip subs.zip
  )
  else
    echo "$0: subs file already downloaded."
fi
