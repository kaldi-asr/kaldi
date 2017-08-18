#!/bin/bash

###########################################################################################
# This script was copied from egs/librispeech/s5/local/data_prep.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Changed wav.scp to use sox to convert and downsample
###########################################################################################

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

# all utterances are FLAC compressed
if ! which sox >&/dev/null; then
   echo "Please install 'sox' on ALL worker nodes!"
   exit 1
fi

spk_file=$src/../SPEAKERS.TXT

mkdir -p $dst || exit 1;

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;
[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1;


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender
utt2dur=$dst/utt2dur; [[ -f "$utt2dur" ]] && rm $utt2dur

for reader_dir in $(find $src -mindepth 1 -maxdepth 1 -type d | sort); do
  reader=$(basename $reader_dir)
  if ! [ $reader -eq $reader ]; then  # not integer.
    echo "$0: unexpected subdirectory name $reader"
    exit 1;
  fi

  reader_gender=$(egrep "^$reader[ ]+\|" $spk_file | awk -F'|' '{gsub(/[ ]+/, ""); print tolower($2)}')
  if [ "$reader_gender" != 'm' ] && [ "$reader_gender" != 'f' ]; then
    echo "Unexpected gender: '$reader_gender'"
    exit 1;
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1;
    fi

    find $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
      awk -v "dir=$chapter_dir" '{printf "%s sox %s/%s.flac -r 8000 -t wavpcm - |\n", $0, dir, $0}' >>$wav_scp|| exit 1

    chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
    [ ! -f  $chapter_trans ] && echo "$0: expected file $chapter_trans to exist" && exit 1
    cat $chapter_trans >>$trans

    # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
    #       to be a different speaker. This is done for simplicity and because we want
    #       e.g. the CMVN to be calculated per-chapter
    awk -v "reader=$reader" -v "chapter=$chapter" '{printf "%s %s-%s\n", $1, reader, chapter}' \
      <$chapter_trans >>$utt2spk || exit 1

    # reader -> gender map (again using per-chapter granularity)
    echo "${reader}-${chapter} $reader_gender" >>$spk2gender
  done
done

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

utils/data/get_utt2dur.sh $dst 1>&2 || exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1;

echo "$0: successfully prepared data in $dst"

exit 0
