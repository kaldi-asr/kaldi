#!/bin/bash 

# Copyright 2014  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0.
# This script processes generates multi-condition training data from clean data dir
# and directory with impulse responses and noises

. ./cmd.sh;
set -e

random_seed=0
num_files_per_job=100
snrs="20:10:15:5:0"
log_dir=exp/make_reverb
max_jobs_run=50
dest_wav_dir=

. ./path.sh;
. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: reverberate_data_dir_speed.sh [options] <src_dir> <impulse-noise-dir> <dest_dir>"
  echo "e.g.:"
  echo " $0 --random-seed 12 data/train_si284 data_multicondition/impulses_noises data/train_si284p"
  exit 1;
fi

src_dir=$1
impnoise_dir=$2
dest_dir=$3

if [ -z $dest_wav_dir ]; then
  dest_wav_dir=$dest_dir/wavs
fi

mkdir -p $dest_dir
mkdir -p $log_dir
mkdir -p $dest_wav_dir

wav_prefix="rev${random_seed}_"
utt_prefix="rev${random_seed}_"
spk_prefix="rev${random_seed}_"
# create the distorted wave files
utils/copy_data_dir.sh --spk-prefix "$spk_prefix" --utt-prefix "$utt_prefix" \
  $src_dir $dest_dir
cat $src_dir/utt2spk | awk -v p=$utt_prefix '{printf("%s%s %s\n", p, $1, $1);}' > $dest_dir/utt2uniq

# create the wav.scp files
cat $src_dir/wav.scp | sed -e "s/^\s*//g" | \
  cut -d' ' -f1 | \
  awk -v p1=$dest_wav_dir -v p2=$wav_prefix \
  '{printf("%s%s%s.wav\n", p1, p2, $1);}'> $log_dir/corrupted_${random_seed}.list

python -c "
import re
file_ids = map(lambda x: x.split()[0], open('$src_dir/wav.scp').readlines())
dest_file_names = map(lambda x: x.split()[0], open('$log_dir/corrupted_${random_seed}.list'))
for file_id, dest_file_name in zip(file_ids, dest_file_names):
  print '$wav_prefix{0} cat {1} |'.format(file_id, dest_file_name)
" > $dest_dir/wav.scp

# modify segments file to point to the new wav files
cat $dest_dir/segments | awk -v p=$wav_prefix \
  '{printf("%s %s%s %s %s\n", $1, p, $2, $3, $4);}' > $log_dir/segments_temp
mv $log_dir/segments_temp $dest_dir/segments

# remove these files as we would have to extract
# features for this new audio and out audio
# is single channel
for file in cmvn.scp feats.scp reco2file_and_channel; do
  rm -f $dest_dir/$file
done

python local/multi_condition/get_reverberate_parameter_lists.py \
  --snrs $snrs --num-files-per-job $num_files_per_job --random-seed $random_seed \
$src_dir/wav.scp $log_dir/corrupted_${random_seed}.list $impnoise_dir \
$log_dir/corrupt_wavs.${random_seed}.list > $log_dir/num_corruption_jobs || exit -1;

num_jobs=$(cat $log_dir/num_corruption_jobs)
$decode_cmd -V --max-jobs-run $max_jobs_run JOB=1:$num_jobs $log_dir/corrupt_wavs.${random_seed}.JOB.log  \
 python local/multi_condition/corrupt.py --temp-file-name $log_dir/temp_JOB.wav $log_dir/corrupt_wavs.${random_seed}.JOB.list || exit 1;

echo "Successfully generated corrupted data and stored it in $dest_dir." && exit 0;
