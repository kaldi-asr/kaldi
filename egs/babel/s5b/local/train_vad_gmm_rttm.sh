#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e 
set -o pipefail

# Begin configuration section.
cmd=run.pl
speech_max_gauss=64
noise_max_gauss=64
sil_max_gauss=32
sil_num_gauss_init=4
noise_num_gauss_init=4
speech_num_gauss_init=4
num_iters=10
stage=-10
cleanup=true
top_frames_threshold=1.0
bottom_frames_threshold=1.0
ignore_energy=true
add_zero_crossing_feats=true
nj=4
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: local/train_vad_gmm_rttm.sh <data> <vad-scp> <segments> <rttm> <exp>"
  echo " e.g.: local/train_vad_gmm_rttm.sh data/dev segments exp/tri4_ali/vad mitfa.rttm exp/vad_dev"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi

data=$1
vad_scp=$2
segments=$3
rttm=$4
dir=$5

mkdir -p $dir

feat_dim=`head -n 1 $data/feats.scp | feat-to-dim scp:- ark,t:- | awk '{print $2}'` || exit 1

ignore_energy_opts=
if $ignore_energy; then
  ignore_energy_opts="select-feats 1-$[feat_dim-1] ark:- ark:- |"
fi

echo "$ignore_energy_opts" > $dir/ignore_energy_opts
echo "$add_zero_crossing_feats" > $dir/add_zero_crossing_feats

for f in $data/feats.scp $data/cmvn.scp $data/utt2spk; do
  [ ! -s $f ] && echo "$0: could not find $f or $f is empty" && exit 1
done 

zc_opts=
[ -f conf/zc_vad.conf ] && zc_opts="--config=conf/zc_vad.conf"
zero_crossing_opts=

if [ $stage -le -3 ]; then
  ###########################################################################
  # Prepare data. 
  # Split the vad in the same way as the data
  ###########################################################################
  rm -rf $dir/data
  utils/copy_data_dir.sh $data $dir/data

  split_data.sh $dir/data $nj || exit 1

  ###########################################################################
  # Add zero-crossing and high-frequency content feats
  ###########################################################################
  if $add_zero_crossing_feats; then
    mkdir -p $dir/data
    $cmd JOB=1:$nj $dir/log/compute_zero_crossing.JOB.log \
      compute-zero-crossings $zc_opts scp:$dir/data/split$nj/JOB/wav.scp ark,scp:$dir/data/zero_crossings.JOB.ark,$dir/data/zero_crossings.JOB.scp || exit 1

    eval cat $dir/data/zero_crossings.{`seq -s',' $nj`}.scp > $dir/data/zero_crossings.scp

    [ ! -f $dir/data/zero_crossings.1.scp ] && exit 1
  fi
fi

###########################################################################
# Get appropriate $feats variable:
# Apply CMVN. Note that we don't apply CMVN to the zero-crossing feats.
# Remove energy from the features. 
# Add zero-crossing feats.
# Add deltas.
###########################################################################
  
feats="ark:apply-cmvn-sliding scp:$dir/data/split$nj/JOB/feats.scp ark:- |${ignore_energy_opts}"

if $add_zero_crossing_feats; then
  feats="${feats}paste-feats ark:- scp:$dir/data/zero_crossings.JOB.scp ark:- |"
fi

feats="${feats}add-deltas ark:- ark:- |"

feats_all="ark:apply-cmvn-sliding scp:$dir/data/feats.scp ark:- |${ignore_energy_opts}"

if $add_zero_crossing_feats; then
  feats_all="${feats_all}paste-feats ark:- scp:$dir/data/zero_crossings.scp ark:- |"
fi

feats_all="${feats_all}add-deltas ark:- ark:- |"

if [ $stage -le -3 ]; then
  $cmd JOB=1:$nj $dir/log/get_speech_segmentation.JOB.log \
    utils/filter_scp.pl -f 2 $dir/data/split$nj/JOB/wav.scp $rttm \| \
    egrep 'LEXEME.*lex' \| \
    awk '{print "SPEAKER "$2" 1 "$4" "$5" <NA> <NA> speech <NA>"}' \| \
    rttmSmooth.pl -s 0 \| diarization/convert_rttm_to_segments.pl \| \
    segmentation-init-from-segments --label=1 - \
    ark:$dir/init_segmentation_speech.JOB.ark

  $cmd $dir/log/get_noise_segmentation.log \
    utils/filter_scp.pl -f 2 $dir/data/wav.scp $rttm \| \
    egrep '"NON-SPEECH|NON-LEX"' \| \
    awk '{print "SPEAKER "$2" 1 "$4" "$5" <NA> <NA> noise <NA>"}' \| \
    rttmSmooth.pl -s 0 \| diarization/convert_rttm_to_segments.pl \| \
    segmentation-init-from-segments --label=2 - \
    ark:$dir/init_segmentation_noise.1.ark

  $cmd $dir/log/get_silence_segmentation.1.log \
    segmentation-init-from-ali scp:$vad_scp ark:- \| \
    segmentation-to-rttm --segments="utils/filter_scp.pl -f 2 $dir/data/wav.scp $segments |" \
    ark:- - \| grep "SILENCE" \| \
    diarization/convert_rttm_to_segments.pl \| \
    segmentation-init-from-segments --label=0 - \
    ark:$dir/init_segmentation_silence.1.ark

  $cmd $dir/log/select_feats_init_noise.log \
    select-feats-from-segmentation --select-label=2 "$feats_all" \
    ark:$dir/init_segmentation_noise.1.ark \
    ark:$dir/init_feats_noise.1.ark || exit 1

  $cmd JOB=1:$nj $dir/log/select_feats_init_speech.JOB.log \
    select-feats-from-segmentation --select-label=1 "$feats" \
    ark:$dir/init_segmentation_speech.JOB.ark \
    ark:$dir/init_feats_speech.JOB.ark || exit 1

  $cmd JOB=1:1 $dir/log/select_feats_init_silence.JOB.log \
    select-feats-from-segmentation --select-label=0 "$feats_all" \
    ark:$dir/init_segmentation_silence.JOB.ark \
    ark:$dir/init_feats_silence.JOB.ark || exit 1
fi

speech_num_gauss=$speech_num_gauss_init
noise_num_gauss=$noise_num_gauss_init
sil_num_gauss=$sil_num_gauss_init

if [ $stage -le -1 ]; then
  $cmd $dir/log/init_gmm_speech.log \
    gmm-global-init-from-feats --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss + 2] \
    "ark:cat $dir/init_feats_speech.{?,??,???}.ark |" $dir/speech.0.mdl || exit 1
  
  $cmd $dir/log/init_gmm_noise.log \
    gmm-global-init-from-feats --num-gauss=$noise_num_gauss --num-iters=$[noise_num_gauss + 2] \
    "ark:cat $dir/init_feats_noise.{?,??,???}.ark |" $dir/noise.0.mdl || exit 1

  $cmd $dir/log/init_gmm_silence.log \
    gmm-global-init-from-feats --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss + 2] \
    "ark:cat $dir/init_feats_silence.{?,??,???}.ark |" $dir/silence.0.mdl || exit 1
fi

x=0
while [ $x -le $num_iters ]; do
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc_gmm_stats_speech.$x.JOB.log \
      gmm-global-acc-stats $dir/speech.$x.mdl \
      "ark:copy-feats ark:$dir/init_feats_speech.JOB.ark ark:- |" \
      $dir/speech_accs.$x.JOB || exit 1
    
    $cmd JOB=1:1 $dir/log/acc_gmm_stats_noise.$x.JOB.log \
      gmm-global-acc-stats $dir/noise.$x.mdl \
      "ark:copy-feats ark:$dir/init_feats_noise.JOB.ark ark:- |" \
      $dir/noise_accs.$x.JOB || exit 1

    $cmd JOB=1:1 $dir/log/acc_gmm_stats_silence.$x.JOB.log \
      gmm-global-acc-stats $dir/silence.$x.mdl \
      "ark:copy-feats ark:$dir/init_feats_silence.JOB.ark ark:- |" \
      $dir/silence_accs.$x.JOB || exit 1

    $cmd $dir/log/gmm_est_speech.$x.log \
      gmm-global-est --mix-up=$speech_num_gauss $dir/speech.$x.mdl \
      "gmm-global-sum-accs - $dir/speech_accs.$x.* |" \
      $dir/speech.$[x+1].mdl || exit 1
    
    $cmd $dir/log/gmm_est_noise.$x.log \
      gmm-global-est --mix-up=$noise_num_gauss $dir/noise.$x.mdl \
      "gmm-global-sum-accs - $dir/noise_accs.$x.* |" \
      $dir/noise.$[x+1].mdl || exit 1

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
  if [ $noise_num_gauss -lt $noise_max_gauss ]; then
    noise_num_gauss=$[noise_num_gauss * 2]
  fi

  x=$[x+1]
done

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

