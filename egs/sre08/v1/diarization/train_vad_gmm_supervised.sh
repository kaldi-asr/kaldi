#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
speech_max_gauss=64
sil_max_gauss=32
sil_num_gauss_init=4
speech_num_gauss_init=4
num_iters=10
stage=-10
cleanup=true
top_frames_threshold=0.1
bottom_frames_threshold=0.2
ignore_energy_opts=
add_zero_crossing_feats=false
nj=4
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: diarization/train_vad_gmm_ntu.sh <data> <vad-dir|vad-scp> <exp>"
  echo " e.g.: diarization/train_vad_gmm_ntu.sh data/dev exp/tri4_ali/vad exp/vad_dev"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi

data=$1
vad_scp=
vad_dir=
if [ -f $2 ]; then
  vad_scp=$2
else 
  vad_dir=$2
fi
dir=$3

mkdir -p $dir

echo "$ignore_energy_opts" > $dir/ignore_energy_opts
echo "$add_zero_crossing_feats" > $dir/add_zero_crossing_feats

for f in $data/feats.scp; do
  [ ! -s $f ] && echo "$0: could not find $f or $f is empty" && exit 1
done 

zc_opts=
[ -f conf/zc_vad.conf ] && zc_opts="--config=conf/zc_vad.conf"

if $add_zero_crossing_feats; then
  mkdir -p $dir/data
  if [ -f $data/segments ]; then
    $cmd JOB=1$nj $dir/log/compute_zero_crossing.JOB.log \
      extract-segments scp:$data/split$nj/JOB/wav.scp $data/split$nj/JOB/segments ark:- \| \
      compute-zero-crossings $zc_opts ark:- ark:$dir/data/zero_crossings.JOB.ark,$dir/data/zero_crossings.JOB.scp || exit 1
  else 
    $cmd JOB=1$nj $dir/log/compute_zero_crossing.JOB.log \
      compute-zero-crossings $zc_opts scp:$data/spilt$nj/JOB/wav.scp ark,scp:$dir/data/zero_crossings.JOB.ark,$dir/data/zero_crossings.JOB.scp || exit 1
  fi

  zero_crossing_opts="paste-feats ark:- scp:$dir/data/zero_crossing_feats.JOB.scp ark:- |"
fi

if [ ! -z "$vad_dir" ]; then

  nj=`cat $vad_dir/num_jobs`
  utils/split_data.sh $data $nj || exit 1

  if [ $stage -le -2 ]; then
    $cmd JOB=1:$nj $dir/log/select_feats_init_speech.JOB.log \
      select-interior-frames "ark:add-deltas scp:$data/split$nj/JOB/feats.scp ark:- |$zero_crossing_opts" \
        "ark:gunzip -c $vad_dir/vad.JOB.ark.gz |" ark:- \| \
        select-top-chunks --frames-proportion=$top_frames_threshold --use-dim-as-weight=0 \
          --window-size=10 ark:- ark:$dir/init_feats_speech.JOB.ark || exit 1

    $cmd JOB=1:$nj $dir/log/select_feats_init_silence.JOB.log \
      select-interior-frames --select-unvoiced-frames=true "ark:add-deltas scp:$data/split$nj/JOB/feats.scp ark:- |$zero_crossing_opts" \
        "ark:gunzip -c $vad_dir/vad.JOB.ark.gz |" ark:- \| \
        select-top-chunks --frames-proportion=$bottom_frames_threshold --use-dim-as-weight=0 --select-bottom-frames=true \
          --window-size=10 ark:- ark:$dir/init_feats_silence.JOB.ark || exit 1
  fi
else
  if [ $stage -le -2 ]; then
    rm -rf $dir/data
    cp -rT $data $dir/data || exit 1
    utils/filter_scp.pl $vad_scp $data/feats.scp > $dir/data/feats.scp || exit 1
    utils/fix_data_dir.sh $dir/data || exit 1

    split_data.sh $dir/data $nj || exit 1
    split_files=
    for n in `seq $nj`; do 
      split_files="$split_files $dir/data/vad.$n.scp"
    done 

    utils/filter_scp.pl $data/utt2spk $vad_scp | split_scp.pl --utt2spk=$data/utt2spk - $split_files || exit 1

    $cmd JOB=1:$nj $dir/log/select_feats_init_speech.JOB.log \
      select-interior-frames "ark:add-deltas scp:$dir/data/split$nj/JOB/feats.scp ark:- |$zero_crossing_feats" \
        scp:$dir/data/vad.JOB.scp ark:- \| \
        select-top-chunks --frames-proportion=$top_frames_threshold --use-dim-as-weight=0 \
          --window-size=10 ark:- ark:$dir/init_feats_speech.JOB.ark || exit 1

    $cmd JOB=1:$nj $dir/log/select_feats_init_silence.JOB.log \
      select-interior-frames --select-unvoiced-frames=true "ark:add-deltas scp:$dir/data/split$nj/JOB/feats.scp ark:- |$zero_crossing_opts" \
        scp:$dir/data/vad.JOB.scp ark:- \| \
        select-top-chunks --frames-proportion=$bottom_frames_threshold --use-dim-as-weight=0 --select-bottom-frames=true \
          --window-size=10 ark:- ark:$dir/init_feats_silence.JOB.ark || exit 1
  fi
fi

speech_num_gauss=$speech_num_gauss_init
sil_num_gauss=$sil_num_gauss_init

if [ $stage -le -1 ]; then
  $cmd $dir/log/init_gmm_speech.log \
    gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss + 2] \
    "ark:cat $dir/init_feats_speech.{?,??,???}.ark |$ignore_energy_opts" $dir/speech.0.mdl || exit 1
  
  $cmd $dir/log/init_gmm_silence.log \
    gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss + 2] \
    "ark:cat $dir/init_feats_silence.{?,??,???}.ark |$ignore_energy_opts" $dir/silence.0.mdl || exit 1
fi

x=0
while [ $x -le $num_iters ]; do
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc_gmm_stats_speech.$x.JOB.log \
      gmm-global-acc-stats $dir/speech.$x.mdl \
      "ark:copy-feats ark:$dir/init_feats_speech.JOB.ark ark:- |$ignore_energy_opts" \
      $dir/speech_accs.$x.JOB || exit 1

    $cmd JOB=1:$nj $dir/log/acc_gmm_stats_silence.$x.JOB.log \
      gmm-global-acc-stats $dir/silence.$x.mdl \
      "ark:copy-feats ark:$dir/init_feats_silence.JOB.ark ark:- |$ignore_energy_opts" \
      $dir/silence_accs.$x.JOB || exit 1

    $cmd $dir/log/gmm_est_speech.$x.log \
      gmm-global-est --mix-up=$speech_num_gauss $dir/speech.$x.mdl \
      "gmm-global-sum-accs - $dir/speech_accs.$x.* |" \
      $dir/speech.$[x+1].mdl || exit 1

    $cmd $dir/log/gmm_est_silence.$x.log \
      gmm-global-est --mix-up=$sil_num_gauss $dir/silence.$x.mdl \
      "gmm-global-sum-accs - $dir/silence_accs.$x.* |" \
      $dir/silence.$[x+1].mdl || exit 1
  fi

  if [ $sil_num_gauss -lt $sil_max_gauss ]; then
    sil_num_gauss=$[sil_num_gauss * 2]
  fi
  if [ $speech_num_gauss -lt $speech_max_gauss ]; then
    speech_num_gauss=$[speech_num_gauss * 2]
  fi

  x=$[x+1]

done

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log


