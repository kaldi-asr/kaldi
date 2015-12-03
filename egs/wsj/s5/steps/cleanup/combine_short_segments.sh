#!/bin/bash

# Copyright 2015  Tom Ko
# Apache 2.0

# Begin configuration section.
frame_shift=10
# End configuration section.

#echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "This script truncates the long audio into smaller overlapping segments"
  echo ""
  echo "Usage: steps/cleanup/combine_short_segments.sh [options] <minimum-length> <data-in> <data-out>"
  echo "e.g.:  steps/cleanup/combine_short_segments.sh 2.0 data/train data/train_min2sec"
  echo ""
  echo "Options:"
  echo "    --frame-shift   <frame-shift|10>         # Frame shift in milliseconds.  Only relevant if"
  echo "                                             # <data-in>/segments does not exist (in which case"
  echo "                                             # the lengths are to be worked out from the features)."
  exit 1;
fi

frame_per_sec=$(( 1000/$frame_shift ))
min_seg_len=$(echo $1*$frame_per_sec | bc)
input_dir=$2
output_dir=$3

for f in spk2utt text utt2spk feats.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

export LC_ALL=C

function check_and_find_match_utt {
  j=$1
  if [ $(echo ${len_list[$j]} "< $min_seg_len" | bc) -eq 1 ]; then
    localmin=100000000
    target=-1
    for k in `seq 0 $(( $uttsno - 1 ))`; do
      sum=$(( ${len_list[$k]} + ${len_list[$j]} ))
      if [ $(echo "$sum >= $min_seg_len" | bc) -eq 1 ] && [ $sum -lt $localmin ] && [ $j != $k ]; then
        target=$k
        localmin=$sum
      fi
    done
    if [ $target -eq -1 ]; then
      echo "$0: no combination of segments has length larger than $min_seg_len"
      exit 1;
    fi

    utt_list[$target]+=${utt_list[$j]}
    ark_list[$target]+=${ark_list[$j]}
    text_list[$target]+=${text_list[$j]}
    len_list[$target]=$localmin
    len_list[$j]=-1
  fi
}

function print_files {
  j=$1
  if [ ${len_list[$j]} -gt ${len_list_bak[$j]} ]; then
    utt=$(echo ${utt_list[$j]} "appended" | sed 's/ /-/g')
    echo $utt $spk >> $output_dir/utt2spk
    echo $utt ${text_list[$j]} >> $output_dir/text
    echo $utt "concat-feats" ${ark_list[$j]} "- |" >> $output_dir/feats.scp
  elif [ ${len_list[$j]} -eq ${len_list_bak[$j]} ]; then
    utt=${utt_list[$j]}
    echo $utt $spk >> $output_dir/utt2spk
    echo $utt ${text_list[$j]} >> $output_dir/text
    echo $utt ${ark_list[$j]} >> $output_dir/feats.scp
  fi
}


mkdir -p $output_dir

if [ -f $input_dir/segments ]; then
  awk -v l=$frame_per_sec '{
    len=($4 - $3) * l;
    if (len > 0) {
      printf("%s %d \n", $1, len);
    }
  }' $input_dir/segments >$output_dir/feats.length
else
  feat-to-len --print-args=false scp:$input_dir/feats.scp ark,t:$output_dir/feats.length
fi

> $output_dir/utt2spk
> $output_dir/feats.scp
> $output_dir/text

for spk in `cat $input_dir/spk2utt | awk '{print $1}'`; do
  echo speaker $i $spk

  echo $spk | utils/filter_scp.pl - $input_dir/spk2utt | awk '{ for(i=2;i<=NF;i++){print $i}}' > $output_dir/uttlist.tmp
  utils/filter_scp.pl $output_dir/uttlist.tmp $input_dir/feats.scp | awk '{print $2}' > $output_dir/ark.tmp
  utils/filter_scp.pl $output_dir/uttlist.tmp $output_dir/feats.length | awk '{print $2}' > $output_dir/length.tmp

  readarray ark_list < $output_dir/ark.tmp
  readarray text_list < <(utils/filter_scp.pl $output_dir/uttlist.tmp $input_dir/text | awk '{$1=""; print $_}') 
  readarray utt_list < $output_dir/uttlist.tmp
  readarray len_list < $output_dir/length.tmp
  readarray len_list_bak < $output_dir/length.tmp

  uttsno=$(wc $output_dir/uttlist.tmp | awk '{print $1}')
  for j in `seq 0 $(( $uttsno - 1 ))`; do
    check_and_find_match_utt $j
  done

  for j in `seq 0 $(( $uttsno - 1 ))`; do
    print_files $j
  done
done
utils/utt2spk_to_spk2utt.pl $output_dir/utt2spk > $output_dir/spk2utt

if [ -f $input_dir/cmvn.scp ]; then
  cp $input_dir/cmvn.scp $output_dir/
fi

rm $output_dir/*.tmp

