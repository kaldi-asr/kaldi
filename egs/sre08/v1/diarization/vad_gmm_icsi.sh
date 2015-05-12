#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e
set -u 
set -o pipefail

stage=-1

## Features paramters
window_size=100                   # 1s

## Phase 1 parameters
num_frames_init_silence=2000      # 20s - Lowest energy frames selected to initialize Silence GMM
num_frames_init_sound=10000       # 100s - Highest energy frames selected to initialize Sound GMM
num_frames_init_sound_next=2000   # 20s - Highest zero crossing frames selected to initialize Sound GMM
sil_num_gauss_init=2
sound_num_gauss_init=2
sil_max_gauss=2
sound_max_gauss=8
sil_gauss_incr=0
sound_gauss_incr=2
sil_frames_incr=2000
sound_frames_incr=10000
sound_frames_next_incr=2000
num_iters=5

## Phase 2 parameters
speech_num_gauss_init=6
sil_max_gauss_phase2=7
sound_max_gauss_phase2=18
speech_max_gauss_phase2=16
sil_gauss_incr_phase2=1
sound_gauss_incr_phase2=2
speech_gauss_incr_phase2=2
num_iters_phase2=5

## Phase 3 parameters
sil_num_gauss_init_phase3=2
speech_num_gauss_init_phase3=2
sil_max_gauss_phase3=5
speech_max_gauss_phase3=12
sil_gauss_incr_phase3=1
speech_gauss_incr_phase3=2
num_iters_phase3=7

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

zc_opts=
[ -f conf/zc_vad.conf ] && zc_opts="--config=conf/zc_vad.conf"

if [ $stage -le -10 ]; then
  if [ ! -f $data/segments ]; then
    compute-zero-crossings $zc_opts scp:$data/wav.scp \
      ark,scp:$dir/zero_crossings.ark,$dir/zero_crossings.scp || exit 1
  else
    compute-zero-crossings $zc_opts "ark:extract-segments scp:$data/wav.scp $data/segments ark:- |" \
      ark,scp:$dir/zero_crossings.ark,$dir/zero_crossings.scp || exit 1
  fi
  extract-column scp:$data/feats.scp ark,scp:$dir/log_energies.ark,$dir/log_energies.scp || { echo "extract-column failed"; exit 1; }
fi

init_model_dir=`dirname $init_speech_model`
ignore_energy_opts=`cat $init_model_dir/ignore_energy_opts` || exit 1

while IFS=$'\n' read line; do
  feats="ark:echo $line | copy-feats scp:- ark:- | add-deltas ark:- ark:- |${ignore_energy_opts}"
  utt_id=$(echo $line | awk '{print $1}')
  echo $utt_id > $dir/$utt_id.list

  sil_num_gauss=$sil_num_gauss_init
  sound_num_gauss=$sound_num_gauss_init
  speech_num_gauss=$speech_num_gauss_init
  num_frames_silence=$num_frames_init_silence
  num_frames_sound=$num_frames_init_sound
  num_frames_sound_next=$num_frames_init_sound_next

  ### Compute likelihoods wrt bootstrapping models
  gmm-global-get-frame-likes $init_speech_model \
    "${feats}" ark:$dir/$utt_id.speech_likes.bootstrap.ark || exit 1

  gmm-global-get-frame-likes $init_silence_model \
    "${feats}" ark:$dir/$utt_id.silence_likes.bootstrap.ark || exit 1

  ### Get bootstrapping VAD
  loglikes-to-class \
    ark:$dir/$utt_id.silence_likes.bootstrap.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.vad.bootstrap.ark || exit 1

  feats="${feats} paste-feats ark:- \"ark:utils/filter_scp.pl $dir/$utt_id.list $dir/zero_crossings.scp | add-deltas scp:- ark:- |\" ark:- |" 
  ### Initialize Silence GMM using lowest energy chunks that were classified
  ### as silence by the bootstrapping model
  select-top-chunks \
    --window-size=$window_size \
    --selection-mask=ark:$tmpdir/$utt_id.vad.bootstrap.ark --select-class=0 \
    --select-bottom-frames=true \
    --weights=scp:$dir/log_energies.scp --num-select-frames=$num_frames_silence \
    "${feats}" ark:- |gmm-global-init-from-feats \
    --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss+2] ark:- \
    $tmpdir/$utt_id.silence.0.mdl || exit 1

  ### Initialize Sound GMM using highest energy and highest zero-crossing 
  ### chunks that were classified
  ### as silence by the bootstrapping model.
  select-top-chunks \
    --window-size=$window_size \
    --selection-mask=ark:$tmpdir/$utt_id.vad.bootstrap.ark --select-class=0 \
    --weights=scp:$dir/log_energies.scp --num-select-frames=$num_frames_sound \
    --weights-next="ark:extract-column scp:$dir/zero_crossings.scp ark:- |" --num-select-frames-next=$num_frames_sound_next \
    "${feats}" ark:- | gmm-global-init-from-feats \
    --num-gauss=$sound_num_gauss --num-iters=$[sound_num_gauss+2] ark:- \
    $tmpdir/$utt_id.sound.0.mdl || exit 1

  ### Compute likelihoods with the newly initialized Silence and Sound GMMs

  gmm-global-get-frame-likes $tmpdir/$utt_id.silence.0.mdl \
    "${feats}" ark:$tmpdir/$utt_id.silence_likes.0.ark || exit 1

  gmm-global-get-frame-likes $tmpdir/$utt_id.sound.0.mdl \
    "${feats}" ark:$tmpdir/$utt_id.sound_likes.0.ark || exit 1

  ### Get initial VAD
  loglikes-to-class \
    ark:$tmpdir/$utt_id.silence_likes.0.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.sound_likes.0.ark ark,t:- | \
    perl -pe 's/\[(.+)]/$1/' | \
    utils/apply_map.pl -f 2- <(echo -e "0 0\n1 1\n2 0") | \
    awk '{printf $1" [ "; for (i = 2; i <= NF; i++) {printf $i" ";}; print "]"}' | \
    copy-vector ark,t:- ark:$tmpdir/$utt_id.vad.init.ark || exit 1

  ### Remove frames that were originally classified as speech 
  ### while training Silence and Sound GMMs
  select-top-chunks \
    --window-size=$window_size \
    --selection-mask=ark:$tmpdir/$utt_id.vad.init.ark --select-class=0 \
    "$feats" ark:$tmpdir/$utt_id.feats.init.ark \
    ark:$tmpdir/$utt_id.mask.init.ark || exit 1

  ## Select energies and zero crossings corresponding to the same selection

  utils/filter_scp.pl $dir/$utt_id.list $dir/log_energies.scp | \
    vector-extract-dims scp:- ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.energies.init.ark || exit 1

  utils/filter_scp.pl $dir/$utt_id.list $dir/zero_crossings.scp | \
    extract-column scp:- ark:- | \
    vector-extract-dims ark:- ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.zc.init.ark || exit 1

  utils/filter_scp.pl $dir/$utt_id.list $tmpdir/$utt_id.vad.init.ark | \
    vector-extract-dims ark,t:- ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.vad.0.ark || exit 1

  vector-extract-dims ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.speech_likes.init.ark || exit 1
  
  x=0
  while [ $x -le $num_iters ]; do
    ### Update Silence GMM using lowest energy chunks currently classified 
    ### as silence  
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

    ### Update Sound GMM using highest energy and highest zero crossing 
    ### chunks currently classified as silence  
    select-top-chunks \
      --window-size=$window_size \
      --selection-mask=ark:$tmpdir/$utt_id.vad.$x.ark --select-class=0 \
      --weights=ark:$tmpdir/$utt_id.energies.init.ark \
      --num-select-frames=$num_frames_sound \
      --weights-next=ark:$tmpdir/$utt_id.zc.init.ark \
      --num-select-frames-next=$num_frames_sound_next \
      ark:$tmpdir/$utt_id.feats.init.ark ark:- | \
      gmm-global-acc-stats \
      $tmpdir/$utt_id.sound.$x.mdl ark:- - | \
      gmm-global-est --mix-up=$sound_num_gauss $tmpdir/$utt_id.sound.$x.mdl \
      - $tmpdir/$utt_id.sound.$[x+1].mdl || exit 1

    ### Compute likelihoods with the current Silence and Sound GMMs
    gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      ark:$tmpdir/$utt_id.feats.init.ark ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark || exit 1

    gmm-global-get-frame-likes $tmpdir/$utt_id.sound.$[x+1].mdl \
      ark:$tmpdir/$utt_id.feats.init.ark ark:$tmpdir/$utt_id.sound_likes.$[x+1].ark || exit 1

    ### Get new VAD predictions on the subset selected for training
    ### Silence and Sound GMMs
    loglikes-to-class \
      ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark \
      ark:$tmpdir/$utt_id.speech_likes.init.ark \
      ark:$tmpdir/$utt_id.sound_likes.$[x+1].ark ark,t:- | \
      perl -pe 's/\[(.+)]/$1/' | \
      utils/apply_map.pl -f 2- <(echo -e "0 0\n1 1\n2 0") | \
      awk '{printf $1" [ "; for (i = 2; i <= NF; i++) {printf $i" ";}; print "]"}' | \
      copy-vector ark,t:- ark:$tmpdir/$utt_id.vad.$[x+1].ark || exit 1
    
    gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      "$feats" ark:$tmpdir/$utt_id.silence_all_likes.$[x+1].ark || exit 1
    
    gmm-global-get-frame-likes $tmpdir/$utt_id.sound.$[x+1].mdl \
      "$feats" ark:$tmpdir/$utt_id.sound_all_likes.$[x+1].ark || exit 1
    
    loglikes-to-class \
      ark:$tmpdir/$utt_id.silence_all_likes.$[x+1].ark \
      ark:$dir/$utt_id.speech_likes.bootstrap.ark \
      ark:$tmpdir/$utt_id.sound_all_likes.$[x+1].ark \
      ark:$tmpdir/$utt_id.pred.$[x+1].ark || exit 1 

    x=$[x+1]
    if [ $sil_num_gauss -lt $sil_max_gauss ]; then
      sil_num_guass=$[sil_num_gauss + sil_gauss_incr]
      num_frames_silence=$[num_frames_silence +  silence_frames_incr]
    fi
    if [ $sound_num_gauss -lt $sound_max_gauss ]; then
      sound_num_gauss=$[sound_num_gauss + sound_gauss_incr]
      num_frames_sound=$[num_frames_sound + sound_frames_incr]
      num_frames_sound_next=$[num_frames_sound_next + sound_frames_next_incr]
    fi
  done    ## Done training Silence and Speech GMMs

  ### Compute likelihoods with the current Silence and Sound GMMs
  gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$x.mdl \
    "$feats" ark:$phase2_dir/$utt_id.silence_likes.init.ark || exit 1

  gmm-global-get-frame-likes $tmpdir/$utt_id.sound.$x.mdl \
    "$feats" ark:$phase2_dir/$utt_id.sound_likes.init.ark || exit 1

  ### Compute initial segmentation for phase 2 training
  loglikes-to-class \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$phase2_dir/$utt_id.sound_likes.init.ark \
    ark:$phase2_dir/$utt_id.seg.init.ark || exit 1

  ### Initialize Speech GMM
  select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase2_dir/$utt_id.seg.init.ark --select-class=1 \
    "$feats" ark:- | gmm-global-init-from-feats \
    --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss+2] \
    ark:- $phase2_dir/$utt_id.speech.0.mdl || exit 1
  
  gmm-global-get-frame-likes $phase2_dir/$utt_id.speech.0.mdl \
    "$feats" ark:$phase2_dir/$utt_id.speech_likes.init.ark || exit 1
  
  ### Compute initial segmentation for phase 2 training
  loglikes-to-class \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$phase2_dir/$utt_id.speech_likes.init.ark \
    ark:$phase2_dir/$utt_id.sound_likes.init.ark \
    ark:$phase2_dir/$utt_id.pred.init.ark || exit 1

  cp $tmpdir/$utt_id.silence.$x.mdl $phase2_dir/$utt_id.silence.0.mdl
  cp $tmpdir/$utt_id.sound.$x.mdl $phase2_dir/$utt_id.sound.0.mdl
  
  #### Update Silence and Sound GMMs using new segmentation
  #select-top-chunks --window-size=1 \
  #  --selection-mask=ark:$phase2_dir/$utt_id.seg.init.ark --select-class=0 \
  #  "$feats" ark:- | gmm-global-acc-stats \
  #  $tmpdir/$utt_id.silence.$x.mdl ark:- - | \
  #  gmm-global-est --mix-up=$sil_num_gauss \
  #  $tmpdir/$utt_id.silence.$x.mdl - $phase2_dir/$utt_id.silence.0.mdl || exit 1

  #select-top-chunks --window-size=1 \
  #  --selection-mask=ark:$phase2_dir/$utt_id.seg.init.ark --select-class=2 \
  #  "$feats" ark:- | gmm-global-acc-stats \
  #  $tmpdir/$utt_id.sound.$x.mdl ark:- - | \
  #  gmm-global-est --mix-up=$sound_num_gauss \
  #  $tmpdir/$utt_id.sound.$x.mdl - $phase2_dir/$utt_id.sound.0.mdl || exit 1

  x=0
  while [ $x -le $num_iters_phase2 ]; do
    ### Compute likelihoods with the current Silence, Speech and Sound GMMs
    gmm-global-get-frame-likes $phase2_dir/$utt_id.silence.$x.mdl \
      "$feats" ark:$phase2_dir/$utt_id.silence_likes.$x.ark || exit 1

    gmm-global-get-frame-likes $phase2_dir/$utt_id.sound.$x.mdl \
      "$feats" ark:$phase2_dir/$utt_id.sound_likes.$x.ark || exit 1

    gmm-global-get-frame-likes $phase2_dir/$utt_id.speech.$x.mdl \
      "$feats" ark:$phase2_dir/$utt_id.speech_likes.$x.ark || exit 1

    ### Get segmentation
    loglikes-to-class \
      ark:$phase2_dir/$utt_id.silence_likes.$x.ark \
      ark:$phase2_dir/$utt_id.speech_likes.$x.ark \
      ark:$phase2_dir/$utt_id.sound_likes.$x.ark \
      ark:$phase2_dir/$utt_id.seg.$x.ark || exit 1

    ### Update Speech GMM
    select-top-chunks --window-size=1 \
      --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=1 \
      "$feats" ark:- | gmm-global-acc-stats \
      $phase2_dir/$utt_id.speech.$x.mdl ark:- - | \
      gmm-global-est --mix-up=$speech_num_gauss \
      $phase2_dir/$utt_id.speech.$x.mdl - $phase2_dir/$utt_id.speech.$[x+1].mdl || exit 1

    ### Update Silence GMM
    select-top-chunks --window-size=1 \
      --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=0 \
      "$feats" ark:- | gmm-global-acc-stats \
      $phase2_dir/$utt_id.silence.$x.mdl ark:- - | \
      gmm-global-est --mix-up=$sil_num_gauss \
      $phase2_dir/$utt_id.silence.$x.mdl - $phase2_dir/$utt_id.silence.$[x+1].mdl || exit 1

    ### Update Sound GMM
    select-top-chunks --window-size=1 \
      --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=2 \
      "$feats" ark:- | gmm-global-acc-stats \
      $phase2_dir/$utt_id.sound.$x.mdl ark:- - | \
      gmm-global-est --mix-up=$sound_num_gauss \
      $phase2_dir/$utt_id.sound.$x.mdl - $phase2_dir/$utt_id.sound.$[x+1].mdl || exit 1

    if [ $sil_num_gauss -lt $sil_max_gauss_phase2 ]; then
      sil_num_gauss=$[sil_num_gauss + sil_gauss_incr_phase2]
    fi

    if [ $sound_num_gauss -lt $sound_max_gauss_phase2 ]; then
      sound_num_gauss=$[sound_num_gauss + sound_gauss_incr_phase2]
    fi

    if [ $speech_num_gauss -lt $speech_max_gauss_phase2 ]; then
      speech_num_gauss=$[speech_num_gauss + speech_gauss_incr_phase2]
    fi

    x=$[x+1]
  done  ## Done training all 3 GMMs

  x=$[x-1]
  copy-vector ark:$phase2_dir/$utt_id.seg.$x.ark ark,t:- | \
    perl -pe 's/\[(.+)]/$1/' | \
    utils/apply_map.pl -f 2- <(echo -e "0 0\n1 1\n2 1") | \
    awk '{printf $1" [ "; for (i = 2; i <= NF; i++) {printf $i" ";}; print "]"}' | \
    copy-vector ark,t:- ark:$phase3_dir/$utt_id.sil_nonsil.$x.ark || exit 1

  select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase3_dir/$utt_id.sil_nonsil.$x.ark --select-class=1 \
    "$feats" ark:- | gmm-global-init-from-feats \
    --num-gauss=$[sound_num_gauss + speech_num_gauss] --num-iters=20 \
    ark:- $phase2_dir/$utt_id.nonsil.$x.mdl || exit 1
   
  gmm-global-get-frame-likes $phase2_dir/$utt_id.silence.$x.mdl \
    "$feats" ark:$phase2_dir/$utt_id.silence_likes.pred.$x.ark || exit 1
  gmm-global-get-frame-likes $phase2_dir/$utt_id.nonsil.$x.mdl \
    "$feats" ark:$phase2_dir/$utt_id.nonsil_likes.pred.$x.ark || exit 1

  loglikes-to-class \
    ark:$phase2_dir/$utt_id.silence_likes.pred.$x.ark \
    ark:$phase2_dir/$utt_id.nonsil_likes.pred.$x.ark \
    ark:$phase2_dir/$utt_id.pred.nonsil.ark || exit 1 

  nonsil_like=$(select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase3_dir/$utt_id.sil_nonsil.$x.ark --select-class=1 \
    "$feats" ark:- | gmm-global-get-frame-likes \
    $phase2_dir/$utt_id.nonsil.$x.mdl ark:- ark,t:- | \
    perl -pe 's/.*\[(.+)]/$1/' | \
    perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum / $i)')

  speech_like=$(select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=1 \
    "$feats" ark:- | gmm-global-get-frame-likes \
    $phase2_dir/$utt_id.speech.$x.mdl ark:- ark,t:- | \
    perl -pe 's/.*\[(.+)]/$1/' | \
    perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum / $i)')

  sound_like=$(select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase2_dir/$utt_id.seg.$x.ark --select-class=2 \
    "$feats" ark:- | gmm-global-get-frame-likes \
    $phase2_dir/$utt_id.sound.$x.mdl ark:- ark,t:- | \
    perl -pe 's/.*\[(.+)]/$1/' | \
    perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum / $i)')

  merge_nonsil=false
  if [ ! -z `perl -e "print \"true\" if ($sound_like + $speech_like < $nonsil_like)"` ]; then
    merge_nonsil=true
  fi

  if $merge_nonsil; then
    speech_num_gauss=$speech_num_gauss_init_phase3
    sil_num_gauss=$sil_num_gauss_init_phase3

    select-top-chunks --window-size=1 \
      --selection-mask=ark:$phase3_dir/$utt_id.sil_nonsil.$x.ark --select-class=1 \
      "$feats" ark:- | gmm-global-init-from-feats \
      --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss+2] \
      ark:- $phase3_dir/$utt_id.speech.0.mdl || exit 1

    select-top-chunks --window-size=1 \
      --selection-mask=ark:$phase3_dir/$utt_id.sil_nonsil.$x.ark --select-class=0 \
      "$feats" ark:- | gmm-global-init-from-feats \
      --num-gauss=$sil_num_gauss --num-iters=$[sil_num_gauss+2] \
      ark:- $phase3_dir/$utt_id.silence.0.mdl || exit 1

    cp $phase2_dir/$utt_id.silence.$x.mdl $phase3_dir/$utt_id.silence.0.mdl || exit 1
    cp $phase2_dir/$utt_id.nonsil.$x.mdl $phase3_dir/$utt_id.speech.0.mdl || exit 1

    x=0
    while [ $x -lt $num_iters_phase3 ]; do
      ### Compute likelihoods with the current Silence and Speech
      gmm-global-get-frame-likes $phase3_dir/$utt_id.silence.$x.mdl \
        "$feats" ark:$phase3_dir/$utt_id.silence_likes.$x.ark || exit 1

      gmm-global-get-frame-likes $phase3_dir/$utt_id.speech.$x.mdl \
        "$feats" ark:$phase3_dir/$utt_id.speech_likes.$x.ark || exit 1

      ### Get current VAD  
      loglikes-to-class \
        ark:$phase3_dir/$utt_id.silence_likes.$x.ark \
        ark:$phase3_dir/$utt_id.speech_likes.$x.ark \
        ark:$phase3_dir/$utt_id.vad.$x.ark || exit 1

      ### Update Speech GMM 
      select-top-chunks --window-size=1 \
        --selection-mask=ark:$phase3_dir/$utt_id.vad.$x.ark --select-class=1 \
        "$feats" ark:- | gmm-global-acc-stats \
        $phase3_dir/$utt_id.speech.$x.mdl ark:- - | \
        gmm-global-est-map \
        $phase3_dir/$utt_id.speech.$x.mdl - $phase3_dir/$utt_id.speech.$[x+1].mdl || exit 1

      ### Update Silence GMM
      select-top-chunks --window-size=1 \
        --selection-mask=ark:$phase3_dir/$utt_id.vad.$x.ark --select-class=0 \
        "$feats" ark:- | gmm-global-acc-stats \
        $phase3_dir/$utt_id.silence.$x.mdl ark:- - | \
        gmm-global-est-map \
        $phase3_dir/$utt_id.silence.$x.mdl - $phase3_dir/$utt_id.silence.$[x+1].mdl || exit 1

      if [ $sil_num_gauss -lt $sil_max_gauss_phase3 ]; then
        sil_num_gauss=$[sil_num_gauss + sil_gauss_incr_phase3]
      fi

      if [ $speech_num_gauss -lt $speech_max_gauss_phase3 ]; then
        speech_num_gauss=$[speech_num_gauss + speech_gauss_incr_phase3]
      fi

      x=$[x+1]
    done

    cp $phase3_dir/$utt_id.silence.$x.mdl $dir/$utt_id.silence.final.mdl
    cp $phase3_dir/$utt_id.speech.$x.mdl $dir/$utt_id.speech.final.mdl
  fi
done < $data/feats.scp
