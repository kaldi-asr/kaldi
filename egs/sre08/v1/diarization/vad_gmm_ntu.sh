#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -u
set -e 
set -o pipefail

stage=-1

## Features paramters
window_size=100                   # 1s
filter_using_zero_crossings=true

## Phase 1 parameters
num_frames_silence_init=2000      # 20s - Lowest energy frames selected to initialize Silence GMM
sil_num_gauss_init=2
sil_max_gauss=2
sil_gauss_incr=0
silence_frames_incr=2000
num_iters=5
min_sil_variance=1
min_speech_variance=0

## Phase 2 parameters
speech_num_gauss_init=6
sil_max_gauss_phase2=7
speech_max_gauss_phase2=16
sil_gauss_incr_phase2=1
speech_gauss_incr_phase2=2
num_iters_phase2=5
window_size_phase2=10

. path.sh
. parse_options.sh || exit 1

if [ $# -ne 4 ]; then
  echo "Usage: vad_gmm_icsi.sh <data> <init-silence-model> <init-speech-model> <dir>"
  echo " e.g.: vad_gmm_icsi.sh data/rt05_eval exp/librispeech_s5/vad_model/silence.0.mdl exp/librispeech_s5/vad_model/speech.0.mdl exp/vad_rt05_eval"
  exit 1
fi

data=$1
init_silence_model=$2
init_speech_model=$3
dir=$4

mkdir -p $dir
tmpdir=$dir/phase1
phase2_dir=$dir/phase2
phase3_dir=$dir/phase3

mkdir -p $tmpdir
mkdir -p $phase2_dir
mkdir -p $phase3_dir

init_model_dir=`dirname $init_speech_model`
ignore_energy_opts=`cat $init_model_dir/ignore_energy_opts` || exit 1
add_zero_crossing_feats=`cat $init_model_dir/add_zero_crossing_feats` || exit 1

zc_opts=
[ -f conf/zc_vad.conf ] && zc_opts="--config=conf/zc_vad.conf"

while IFS=$'\n' read line; do
  feats="ark:echo $line | copy-feats scp:- ark:- | add-deltas ark:- ark:- |${ignore_energy_opts}"
  utt_id=$(echo $line | awk '{print $1}')
  echo $utt_id > $dir/$utt_id.list

  if $add_zero_crossing_feats || $filter_using_zero_crossings; then
    if [ -f $data/segments ]; then
      utils/filter_scp.pl $dir/$utt_id.list $data/segments | \
        extract-segments scp:$data/wav.scp - ark:- | \
        compute-zero-crossings $zc_opts ark:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    else 
      utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp | \
        compute-zero-crossings $zc_opts scp:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    fi
  fi

  extract-column "scp:utils/filter_scp.pl $dir/$utt_id.list $data/feats.scp |" \
    ark:$dir/$utt_id.log_energies.ark || exit 1

  sil_num_gauss=$sil_num_gauss_init
  speech_num_gauss=$speech_num_gauss_init
  num_frames_silence=$num_frames_silence_init
  
  if $add_zero_crossing_feats; then
    feats="${feats} paste-feats ark:- \"ark:add-deltas ark:$dir/$utt_id.zero_crossings.ark ark:- |\" ark:- |" 
  fi

  ### Compute likelihoods wrt bootstrapping models
  gmm-global-get-frame-likes $init_speech_model \
    "${feats}" ark:$dir/$utt_id.speech_likes.bootstrap.ark || exit 1

  gmm-global-get-frame-likes $init_silence_model \
    "${feats}" ark:$dir/$utt_id.silence_likes.bootstrap.ark || exit 1
  
  ### Get bootstrapping VAD
  loglikes-to-class --verbose=2 --weights=ark:$dir/$utt_id.post.bootstrap.ark \
    ark:$dir/$utt_id.silence_likes.bootstrap.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.vad.bootstrap.ark || exit 1
  
  ### Initialize Silence GMM using lowest energy chunks that were classified
  ### as silence by the bootstrapping model
  if ! $filter_using_zero_crossings; then
    select-top-chunks \
      --window-size=$window_size \
      --selection-mask=ark:$tmpdir/$utt_id.vad.bootstrap.ark --select-class=0 \
      --select-bottom-frames=true \
      --weights=ark:$dir/$utt_id.log_energies.ark --num-select-frames=$num_frames_silence \
      "${feats}" ark:- ark:$tmpdir/$utt_id.mask.0.ark | gmm-global-init-from-feats \
      --min-variance=$min_sil_variance --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss] ark:- \
      $tmpdir/$utt_id.silence.0.mdl || exit 1
  else
    select-top-chunks \
      --window-size=$window_size \
      --selection-mask=ark:$tmpdir/$utt_id.vad.bootstrap.ark --select-class=0 \
      --weights=ark:$dir/$utt_id.zero_crossings.ark --num-select-frames=$num_frames_silence \
      "${feats}" ark:- ark:$tmpdir/$utt_id.mask.0.ark | gmm-global-init-from-feats \
      --min-variance=$min_sil_variance --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss] ark:- \
      $tmpdir/$utt_id.silence.0.mdl || exit 1
  fi

  gmm-global-get-frame-likes $tmpdir/$utt_id.silence.0.mdl \
    "${feats}" ark:$tmpdir/$utt_id.silence_likes.0.ark || exit 1
  
  ### Get initial VAD
  loglikes-to-class --verbose=2 --weights=ark:$tmpdir/$utt_id.post.init.ark \
    ark:$tmpdir/$utt_id.silence_likes.0.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.vad.init.ark || exit 1

  ### Remove frames that were originally classified as speech 
  ### while training Silence and Sound GMMs
  select-top-chunks \
    --window-size=$window_size \
    --selection-mask=ark:$tmpdir/$utt_id.vad.bootstrap.ark --select-class=0 \
    "$feats" ark:$tmpdir/$utt_id.feats.init.ark \
    ark:$tmpdir/$utt_id.mask.init.ark || exit 1
  
  #select-top-chunks \
  #  --window-size=$window_size \
  #  --selection-mask=ark:$tmpdir/$utt_id.vad.init.ark --select-class=0 \
  #  "$feats" ark:$tmpdir/$utt_id.feats.init.ark \
  #  ark:$tmpdir/$utt_id.mask.init.ark || exit 1
  
  ## Select energies and zero crossings corresponding to the same selection

  vector-extract-dims ark:$dir/$utt_id.log_energies.ark \
    ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.energies.init.ark || exit 1
  
  vector-extract-dims ark:$tmpdir/$utt_id.vad.init.ark \
    ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.vad.0.ark || exit 1
  
  vector-extract-dims \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.speech_likes.init.ark || exit 1

  x=0
  while [ $x -le $num_iters ]; do
    ### Update Silence GMM using lowest energy chunks currently classified 
    ### as silence  

    if ! $filter_using_zero_crossings; then
      select-top-chunks \
        --window-size=$window_size \
        --selection-mask=ark:$tmpdir/$utt_id.vad.$x.ark --select-class=0 \
        --select-bottom-frames=true --weights=ark:$tmpdir/$utt_id.energies.init.ark \
        --num-select-frames=$num_frames_silence \
        ark:$tmpdir/$utt_id.feats.init.ark ark:- | \
        gmm-global-acc-stats \
        $tmpdir/$utt_id.silence.$x.mdl ark:- - | \
        gmm-global-est --mix-up=$sil_num_gauss $tmpdir/$utt_id.silence.$x.mdl \
        - $tmpdir/$utt_id.silence.$[x+1].mdl || exit 1
    else
      select-top-chunks \
        --window-size=$window_size \
        --selection-mask=ark:$tmpdir/$utt_id.vad.$x.ark --select-class=0 \
        --weights=ark:$tmpdir/$utt_id.zero_crossings.init.ark \
        --num-select-frames=$num_frames_silence \
        ark:$tmpdir/$utt_id.feats.init.ark ark:- | \
        gmm-global-acc-stats \
        $tmpdir/$utt_id.silence.$x.mdl ark:- - | \
        gmm-global-est --mix-up=$sil_num_gauss $tmpdir/$utt_id.silence.$x.mdl \
        - $tmpdir/$utt_id.silence.$[x+1].mdl || exit 1
    fi

    ### Compute likelihoods with the current Silence and Sound GMMs
    gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      ark:$tmpdir/$utt_id.feats.init.ark ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark || exit 1

    ### Get new VAD predictions on the subset selected for training
    ### Silence and Sound GMMs
    loglikes-to-class --verbose=2 --weights=ark:$tmpdir/$utt_id.post.$[x+1].ark \
      ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark \
      ark:$tmpdir/$utt_id.speech_likes.init.ark \
      ark:$tmpdir/$utt_id.vad.$[x+1].ark || exit 1
    
    gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      "$feats" ark:- | \
      loglikes-to-class --verbose=2 --weights=ark:$tmpdir/$utt_id.pred_post.$[x+1].ark ark:- \
      ark:$dir/$utt_id.speech_likes.bootstrap.ark \
      ark:$tmpdir/$utt_id.pred.$[x+1].ark || exit 1 

    x=$[x+1]
    if [ $sil_num_gauss -lt $sil_max_gauss ]; then
      sil_num_guass=$[sil_num_gauss + sil_gauss_incr]
      num_frames_silence=$[num_frames_silence +  silence_frames_incr]
    fi
  done    ## Done training Silence and Speech GMMs
  
  ### Compute likelihoods with the current Silence and Sound GMMs
  gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$x.mdl \
    "$feats" ark:$phase2_dir/$utt_id.silence_likes.init.ark || exit 1
  
  ### Compute initial segmentation for phase 2 training
  loglikes-to-class --verbose=2 --weights=ark:$phase2_dir/$utt_id.post.init.ark \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$phase2_dir/$utt_id.seg.init.ark || exit 1

  ### Initialize Speech GMM
  select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase2_dir/$utt_id.seg.init.ark --select-class=1 \
    "$feats" ark:- | gmm-global-init-from-feats --min-variance=$min_speech_variance \
    --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss+2] \
    ark:- $phase2_dir/$utt_id.speech.0.mdl || exit 1
    
  gmm-global-get-frame-likes $phase2_dir/$utt_id.speech.0.mdl \
    "$feats" ark:$phase2_dir/$utt_id.speech_likes.init.ark || exit 1
  
  loglikes-to-class --verbose=2 --weights=ark:$phase2_dir/$utt_id.pred.post.init.ark \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$phase2_dir/$utt_id.speech_likes.init.ark \
    ark:$phase2_dir/$utt_id.pred.init.ark || exit 1

  cp $tmpdir/$utt_id.silence.$x.mdl $phase2_dir/$utt_id.silence.0.mdl || exit 1

  x=0
  while [ $x -le $num_iters_phase2 ]; do
    ### Compute likelihoods with the current Silence, Speech and Sound GMMs
    gmm-global-get-frame-likes $phase2_dir/$utt_id.silence.$x.mdl \
      "$feats" ark:$phase2_dir/$utt_id.silence_likes.$x.ark || exit 1

    gmm-global-get-frame-likes $phase2_dir/$utt_id.speech.$x.mdl \
      "$feats" ark:$phase2_dir/$utt_id.speech_likes.$x.ark || exit 1

    ### Get segmentation
    loglikes-to-class --verbose=2 --weights=ark:$phase2_dir/$utt_id.pred.$x.ark \
      ark:$phase2_dir/$utt_id.silence_likes.$x.ark \
      ark:$phase2_dir/$utt_id.speech_likes.$x.ark \
      ark:$phase2_dir/$utt_id.seg.$x.ark || exit 1

    ### Update Speech GMM
    select-top-chunks --window-size=$window_size_phase2 \
      --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=1 \
      "$feats" ark:- | gmm-global-acc-stats \
      $phase2_dir/$utt_id.speech.$x.mdl ark:- - | \
      gmm-global-est --mix-up=$speech_num_gauss \
      $phase2_dir/$utt_id.speech.$x.mdl - $phase2_dir/$utt_id.speech.$[x+1].mdl || exit 1

    cp $phase2_dir/$utt_id.silence.$x.mdl $phase2_dir/$utt_id.silence.$[x+1].mdl
    ### Update Silence GMM
    #select-top-chunks --window-size=$window_size_phase2 \
    #  --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=0 \
    #  "$feats" ark:- | gmm-global-acc-stats \
    #  $phase2_dir/$utt_id.silence.$x.mdl ark:- - | \
    #  gmm-global-est --mix-up=$sil_num_gauss --min-gaussian-occupancy=100 \
    #  $phase2_dir/$utt_id.silence.$x.mdl - $phase2_dir/$utt_id.silence.$[x+1].mdl || exit 1

    if [ $sil_num_gauss -lt $sil_max_gauss_phase2 ]; then
      sil_num_gauss=$[sil_num_gauss + sil_gauss_incr_phase2]
    fi

    if [ $speech_num_gauss -lt $speech_max_gauss_phase2 ]; then
      speech_num_gauss=$[speech_num_gauss + speech_gauss_incr_phase2]
    fi

    x=$[x+1]
  done  ## Done training all 3 GMMs
    
  cp $phase2_dir/$utt_id.silence.$x.mdl $dir/$utt_id.silence.final.mdl
  cp $phase2_dir/$utt_id.speech.$x.mdl $dir/$utt_id.speech.final.mdl

done < $data/feats.scp
