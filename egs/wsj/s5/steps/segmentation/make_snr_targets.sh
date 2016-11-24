#!/bin/bash 

# Copyright 2015-16  Vimal Manohar
# Apache 2.0
set -e 
set -o pipefail

nj=4
cmd=run.pl
stage=0

data_id=

compress=true
target_type=Irm
apply_exp=false

ali_rspecifier=
silence_phones_str=0

ignore_noise_dir=false

ceiling=inf
floor=-inf

length_tolerance=2
transform_matrix=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 [options] --target-type (Irm|Snr) <clean-data> <noise-data> <out-data> <log-dir> <path-to-targets>";
  echo " or  : $0 [options] --target-type FbankMask <clean-data> <corrupted-data> <out-data> <log-dir> <path-to-targets>";
  echo "e.g.: $0 data/train_clean_fbank data/train_noise_fbank data/train_corrupted_hires exp/make_snr_targets/train snr_targets"
  echo "options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

clean_data=$1
noise_or_noisy_data=$2
data=$3
tmpdir=$4
targets_dir=$5

mkdir -p $targets_dir 

[ -z "$data_id" ] && data_id=`basename $data`

utils/split_data.sh $clean_data $nj

for n in `seq $nj`; do 
  utils/subset_data_dir.sh --utt-list $clean_data/split$nj/$n/utt2spk $noise_or_noisy_data $noise_or_noisy_data/subset${nj}/$n
done

$ignore_noise_dir && utils/split_data.sh $data $nj

targets_dir=`perl -e '($data,$pwd)= @ARGV; if($data!~m:^/:) { $data = "$pwd/$data"; } print $data; ' $targets_dir ${PWD}`

for n in `seq $nj`; do 
  utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
done

apply_exp_opts=
if $apply_exp; then
  apply_exp_opts=" copy-matrix --apply-exp=true ark:- ark:- |"
fi

copy_feats_opts="copy-feats"
if [ ! -z "$transform_matrix" ]; then
  copy_feats_opts="transform-feats $transform_matrix"
fi

if [ $stage -le 1 ]; then
  if ! $ignore_noise_dir; then
    $cmd JOB=1:$nj $tmpdir/make_`basename $targets_dir`_${data_id}.JOB.log \
      compute-snr-targets --length-tolerance=$length_tolerance --target-type=$target_type \
      ${ali_rspecifier:+--ali-rspecifier="$ali_rspecifier" --silence-phones=$silence_phones_str} \
      --floor=$floor --ceiling=$ceiling \
      "ark:$copy_feats_opts scp:$clean_data/split$nj/JOB/feats.scp ark:- |" \
      "ark,s,cs:$copy_feats_opts scp:$noise_or_noisy_data/subset$nj/JOB/feats.scp ark:- |" \
      ark:- \|$apply_exp_opts \
      copy-feats --compress=$compress ark:- \
      ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp || exit 1
  else
    feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1
    $cmd JOB=1:$nj $tmpdir/make_`basename $targets_dir`_${data_id}.JOB.log \
      compute-snr-targets --length-tolerance=$length_tolerance --target-type=$target_type \
      ${ali_rspecifier:+--ali-rspecifier="$ali_rspecifier" --silence-phones=$silence_phones_str} \
      --floor=$floor --ceiling=$ceiling --binary-targets --target-dim=$feat_dim \
      scp:$data/split$nj/JOB/feats.scp \
      ark:- \|$apply_exp_opts \
      copy-feats --compress=$compress ark:- \
      ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp || exit 1
  fi
fi

for n in `seq $nj`; do
  cat $targets_dir/${data_id}.$n.scp
done > $data/`basename $targets_dir`.scp
