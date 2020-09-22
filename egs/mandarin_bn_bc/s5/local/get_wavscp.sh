#!/bin/bash

. utils/parse_options.sh
. ./path.sh

sph_flag=true
audio_list=$1
datadir=$2


if [ $sph_flag ]; then
  sph2pipe_path=$KALDI_ROOT/tools/sph2pipe_v2.5
  if [ ! -d $sph2pipe ]; then
  	echo "You did not have sph2pipe installed at $KALDI_ROOT/tools"
  	echo "If you do not have SPH format audio files, you can set \"sph_flag\" to false and re-run this script;"
  	echo "otherwise, please go to http://www.openslr.org/3 to download the sph2pipe_v2.5.tar.gz file, "
		echo "unzip it, and put the unzipped folder to $KALDI_ROOT/tools"
		exit 1;
  fi
fi

if [ ! -d $datadir ]; then
  mkdir -p $datadir || { echo "mkdir $datadir failed";  exit 1; }
fi

[ ! -f $datadir/wav.scp ] || rm $datadir/wav.scp || exit 1

while read -r line; do
  filename=$(basename "$line")
  fname="${filename%.*}"
  ext="${filename##*.}"
  ext=$(echo $ext | tr '[:upper:]' '[:lower:]')
  if [ $ext == "wav" ] || [ $ext == "flac" ]; then
    scp_line="$fname sox $line -r 16000 -t wav - | "
	elif [ $ext == "sph" ]; then
		scp_line="$fname $sph2pipe_path/sph2pipe -f wav -p -c 1 $line | sox - -r 16000 -t wav - |"
	else
		echo "The audio format needs to be one of \"wav\", \"sph\", \"flac\""
		exit 1
	fi
	echo $scp_line >> $datadir/wav.scp
done < $audio_list
