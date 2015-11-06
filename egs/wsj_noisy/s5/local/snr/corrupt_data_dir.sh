#!/bin/bash 

# Copyright 2014  Johns Hopkins University (Author: Vijayaditya Peddinti)
#           2015  Tom Ko
#           2015  Vimal Manohar
# Apache 2.0.
# This script processes generates multi-condition training data from clean data dir
# and directory with impulse responses and noises

. ./cmd.sh;
set -e
set -o pipefail

stage=0
random_seed=0
num_files_per_job=100
snrs="20:10:15:5:0:-5"
signal_dbs="5:2:0:-2:-5:-10:-20:-30:-40:-50:-60"
tmp_dir=exp/make_corrupt
output_clean_dir=
output_clean_wav_dir=
output_noise_dir=
output_noise_wav_dir=
dest_wav_dir=
select_only_corruption_with_noise=false
nj=200

. ./path.sh;
. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <src_dir> <impulse-noise-dir> <dest_dir>"
  echo "e.g.:"
  echo " $0 --random-seed 12 data/train_si284 data_multicondition/impulses_noises data/train_si284p"
  exit 1;
fi

src_dir=$1
impnoise_dir=$2
dest_dir=$3

# $impnoise_dir must contain a directory info which has the following files
# impulse_files - list of impulse response wav files
# noise_files - list of noise wav files
# noise_impulse_* - containes pairs of impulse responses and noise files in 
# the following format
# noise_files =<list of noise_wav_files>
# impulse_files =<list of impulse_response_files>

data_id=`basename $src_dir`
if [ -z "$dest_wav_dir" ]; then
  dest_wav_dir=$dest_dir/wavs_${data_id}
fi

dest_wav_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dest_wav_dir ${PWD}`

if [ ! -z "$output_clean_dir" ]; then
  [ -z "$output_clean_wav_dir" ] && output_clean_wav_dir=$output_clean_dir/wavs_${data_id}
  mkdir -p $output_clean_dir $output_clean_wav_dir
  output_clean_wav_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $output_clean_wav_dir ${PWD}`
fi

if [ ! -z "$output_noise_dir" ]; then
  [ -z "$output_noise_wav_dir" ] && output_noise_wav_dir=$output_noise_dir/wavs_${data_id}
  mkdir -p $output_noise_wav_dir
  output_noise_wav_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $output_noise_wav_dir ${PWD}`
fi

mkdir -p $dest_dir
mkdir -p $tmp_dir
mkdir -p $dest_wav_dir

wav_prefix="corrupted${random_seed}_"
utt_prefix="corrupted${random_seed}_"
spk_prefix="corrupted${random_seed}_"

if [ $stage -le 0 ]; then
  # Create the distorted wave files
  utils/copy_data_dir.sh --spk-prefix "$spk_prefix" --utt-prefix "$utt_prefix" \
    $src_dir $dest_dir
  cat $src_dir/utt2spk | \
    awk -v p=$utt_prefix '{printf("%s%s %s\n", p, $1, $1);}' > $dest_dir/utt2uniq

  cat $src_dir/wav.scp | sed -e "s/^\s*//g" | \
    cut -d' ' -f1 | \
    awk -v p1=$dest_wav_dir -v p2=$wav_prefix \
    '{printf("%s%s %s/%s%s.wav\n", p2, $1, p1, p2, $1);}'> $tmp_dir/corrupted_${random_seed}.list

  if [ ! -z "$output_clean_dir" ]; then
    utils/copy_data_dir.sh --extra-files utt2uniq $dest_dir $output_clean_dir
    
    cat $src_dir/wav.scp | sed -e "s/^\s*//g" | \
      cut -d' ' -f1 | \
      awk -v p1=$output_clean_wav_dir -v p2=$wav_prefix \
      '{printf("%s%s %s/%s%s.wav\n", p2, $1, p1, p2, $1);}'> $tmp_dir/clean_${random_seed}.list
  fi
  
  if [ ! -z "$output_noise_dir" ]; then
    utils/copy_data_dir.sh --extra-files utt2uniq $dest_dir $output_noise_dir
    
    cat $src_dir/wav.scp | sed -e "s/^\s*//g" | \
      cut -d' ' -f1 | \
      awk -v p1=$output_noise_wav_dir -v p2=$wav_prefix \
      '{printf("%s%s %s/%s%s.wav\n", p2, $1, p1, p2, $1);}'> $tmp_dir/noise_${random_seed}.list
  fi
fi


## Create a list of new wave files
#if [ $stage -le 1 ]; then
#  # Create the new wav.scp file
#  python -c "
#import re
#file_ids = map(lambda x: x.split()[0], open('$src_dir/wav.scp').readlines())
#dest_file_names = map(lambda x: x.split()[0], open('$tmp_dir/corrupted_${random_seed}.list'))
#for file_id, dest_file_name in zip(file_ids, dest_file_names):
#  print '$wav_prefix{0} cat {1} |'.format(file_id, dest_file_name)
#" > $dest_dir/wav.scp
#
#  if [ ! -z "$output_clean_dir" ]; then
#    python -c "
#import re
#file_ids = map(lambda x: x.split()[0], open('$src_dir/wav.scp').readlines())
#dest_file_names = map(lambda x: x.split()[0], open('$tmp_dir/clean_${random_seed}.list'))
#for file_id, dest_file_name in zip(file_ids, dest_file_names):
#  print '$wav_prefix{0} cat {1} |'.format(file_id, dest_file_name)
#" > $output_clean_dir/wav.scp
#  fi
#  
#  if [ ! -z "$output_noise_dir" ]; then
#    python -c "
#import re
#file_ids = map(lambda x: x.split()[0], open('$src_dir/wav.scp').readlines())
#dest_file_names = map(lambda x: x.split()[0], open('$tmp_dir/noise_${random_seed}.list'))
#for file_id, dest_file_name in zip(file_ids, dest_file_names):
#  print '$wav_prefix{0} cat {1} |'.format(file_id, dest_file_name)
#" > $output_noise_dir/wav.scp
#  fi
#fi

if [ $stage -le 1 ]; then
  # Modify segments file to point to the new wav files
  if [ -f $dest_dir/segments ]; then
    cat $dest_dir/segments | awk -v p=$wav_prefix \
      '{printf("%s %s%s %s %s\n", $1, p, $2, $3, $4);}' > $tmp_dir/segments_temp
    mv $tmp_dir/segments_temp $dest_dir/segments
    
    if [ ! -z "$output_clean_dir" ]; then
      cat $output_clean_dir/segments | awk -v p=$wav_prefix \
        '{printf("%s %s%s %s %s\n", $1, p, $2, $3, $4);}' > $tmp_dir/segments_temp
      mv $tmp_dir/segments_temp $output_clean_dir/segments
    fi
    
    if [ ! -z "$output_noise_dir" ]; then
      cat $output_noise_dir/segments | awk -v p=$wav_prefix \
        '{printf("%s %s%s %s %s\n", $1, p, $2, $3, $4);}' > $tmp_dir/segments_temp
      mv $tmp_dir/segments_temp $output_noise_dir/segments
    fi
  fi
fi

# Remove these files as we would have to extract
# features for this new audio and out audio
# is single channel
for file in cmvn.scp feats.scp reco2file_and_channel; do
  rm -f $dest_dir/$file
  if [ ! -z "$output_clean_dir" ]; then
    rm -f $output_clean_dir/$file
  fi
  if [ ! -z "$output_noise_dir" ]; then
    rm -f $output_noise_dir/$file
  fi
done

if $select_only_corruption_with_noise; then
  if [ $stage -le 2 ]; then
    mkdir -p ${impnoise_dir}_noisy/info
    cp ${impnoise_dir}/info/* ${impnoise_dir}_noisy/info
    cat ${impnoise_dir}/info/noise_impulse_* | grep "impulse_files" | \
      python -c "
import sys
for line in sys.stdin.readlines():
  for x in line.strip().split('=')[1].split():
    print (x)
" > ${impnoise_dir}_noisy/info/impulse_files
  fi
  impnoise_dir=${impnoise_dir}_noisy
fi
[ ! -s $impnoise_dir/info/impulse_files ] && echo "$0: $impnoise_dir/info/impulse_files contains no impulses" && exit 1
[ ! -s $impnoise_dir/info/noise_files ] && echo "$0: $impnoise_dir/info/noise_files contains no noises" && exit 1

sox -n -r 16000 -c 1 $tmp_dir/silence.wav trim 0.0 2.0

cat $src_dir/wav.scp | \
  awk -v sil=$tmp_dir/silence.wav '{
if ($NF == 2) { 
  print $1" cat "$2" | sox "sil" -t wav - "sil" -t wav - |"; 
} else { 
  print $0" sox "sil" -t wav - "sil" -t wav - |"; 
} }' > $tmp_dir/src_wav.scp

if [ $stage -le 3 ]; then
  python local/snr/corrupt_wavs.py \
    --snrs $snrs --signal-dbs $signal_dbs --random-seed $random_seed \
    --output-clean-wav-file-list $tmp_dir/clean_${random_seed}.list \
    --output-noise-wav-file-list $tmp_dir/noise_${random_seed}.list \
    $tmp_dir/src_wav.scp $tmp_dir/corrupted_${random_seed}.list $impnoise_dir \
    $tmp_dir/corrupt_wav_commands.${random_seed}.list
fi

if [ $stage -le 4 ]; then
  corrupt_wav_command_lists=
  for i in `seq $nj`; do
    corrupt_wav_command_lists="$corrupt_wav_command_lists $tmp_dir/corrupt_wav_commands.${random_seed}.$i.list"
  done

  if $select_only_corruptions_with_noise; then
    grep "noise-file=" $tmp_dir/corrupt_wav_commands.${random_seed}.list | \
      awk '{print $1}' > $tmp_dir/wavs_with_noise.${random_seed}.list

    utils/filter_scp.pl $tmp_dir/wavs_with_noise.${random_seed}.list \
      $tmp_dir/corrupt_wav_commands.${random_seed}.list | \
      utils/split_scp.pl /dev/stdin \
      $corrupt_wav_command_lists

    utils/filter_scp.pl $tmp_dir/wavs_with_noise.${random_seed}.list \
      $tmp_dir/corrupted_${random_seed}.list | sort -k1,1 -u > $dest_dir/wav.scp
    utils/filter_scp.pl $tmp_dir/wavs_with_noise.${random_seed}.list \
      $tmp_dir/clean_${random_seed}.list | sort -k1,2 -u > $output_clean_dir/wav.scp
    utils/filter_scp.pl $tmp_dir/wavs_with_noise.${random_seed}.list \
      $tmp_dir/noise_${random_seed}.list | sort -k1,2 -u > $output_noise_dir/wav.scp

    utils/fix_data_dir.sh $dest_dir
    utils/fix_data_dir.sh $output_noise_dir
    utils/fix_data_dir.sh $output_clean_dir
  else
    utils/split_scp.pl $tmp_dir/wavs_with_noise.${random_seed}.list \
      $corrupt_wav_command_lists
  fi


  $train_cmd JOB=1:$nj $tmp_dir/corrupt_wavs.${random_seed}.JOB.log \
    cat $tmp_dir/corrupt_wav_commands.${random_seed}.JOB.list \| \
    awk '{a=""; for (i=2; i<=NF; i++) a=a" "$i; print(a)}' \| \
    bash -xe || exit 1
fi

echo "Successfully generated corrupted data and stored it in $dest_dir." && exit 0;
