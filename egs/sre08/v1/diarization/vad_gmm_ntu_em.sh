#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -u
set -e 
set -o pipefail

stage=-1

## Features paramters
ignore_energy_opts=               # select-feats 1-12,14-25,27,38
window_size=100                   # 1s

## Phase 1 parameters
num_frames_init_silence=2000      # 20s - Lowest energy frames selected to initialize Silence GMM
sil_num_gauss_init=2
sil_max_gauss=2
sil_gauss_incr=0
sil_frames_incr=2000
num_iters=5

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

# Prepare a lang directory
if [ $stage -le -12 ]; then
  mkdir -p $dir/local
  mkdir -p $dir/local/dict
  mkdir -p $dir/local/lm

  echo "1" > $dir/local/dict/silence_phones.txt
  echo "1" > $dir/local/dict/optional_silence.txt
  echo "2" > $dir/local/dict/nonsilence_phones.txt
  echo -e "1 1\n2 2" > $dir/local/dict/lexicon.txt
  echo -e "1\n2\n1 2" > $dir/local/dict/extra_questions.txt

  mkdir -p $dir/lang
  diarization/prepare_vad_lang.sh --num-sil-states 1 --num-nonsil-states 1 \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  fstisstochastic $dir/lang/G.fst  || echo "[info]: G not stochastic."
  diarization/prepare_vad_lang.sh --num-sil-states 30 --num-nonsil-states 75 \
    $dir/local/dict $dir/local/lang $dir/lang_test || exit 1
fi

feat_dim=`feat-to-dim "ark:head -n 1 $data/feats.scp | copy-feats scp:- ark:- | add-deltas ark:- ark:- |${ignore_energy_opts}" ark,t:- | awk '{print $2}'` || exit 1

if [ $stage -le -11 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    --binary=false $dir/lang_test/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans_test.mdl || exit 1
  
  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diarization/make_vad_graph.sh --iter trans_test $dir/lang_test $dir $dir/graph_test || exit 1
fi
 
cat <<EOF > $dir/pdf_to_tid.map
0 1
1 3
EOF


if [ $stage -le -10 ]; then
  if [ ! -f $data/segments ]; then
    compute-zero-crossings --write-as-vector=true scp:$data/wav.scp \
      ark,scp:$dir/zero_crossings.ark,$dir/zero_crossings.scp || exit 1
  else
    compute-zero-crossings --write-as-vector=true "ark:extract-segments scp:$data/wav.scp $data/segments ark:- |" \
      ark,scp:$dir/zero_crossings.ark,$dir/zero_crossings.scp || exit 1
  fi
  extract-column scp:$data/feats.scp ark,scp:$dir/log_energies.ark,$dir/log_energies.scp || { echo "extract-column failed"; exit 1; }
fi


while IFS=$'\n' read line; do
  feats="ark:echo $line | copy-feats scp:- ark:- | add-deltas ark:- ark:- |${ignore_energy_opts}"
  utt_id=$(echo $line | awk '{print $1}')
  echo $utt_id > $dir/$utt_id.list

  sil_num_gauss=$sil_num_gauss_init
  speech_num_gauss=$speech_num_gauss_init
  num_frames_silence=$num_frames_init_silence

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

  gmm-global-get-frame-likes $tmpdir/$utt_id.silence.0.mdl \
    "${feats}" ark:$tmpdir/$utt_id.silence_likes.0.ark || exit 1
  
  ### Get initial VAD
  loglikes-to-class \
    ark:$tmpdir/$utt_id.silence_likes.0.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$tmpdir/$utt_id.vad.init.ark || exit 1

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
  
  utils/filter_scp.pl $dir/$utt_id.list $tmpdir/$utt_id.vad.init.ark | \
    vector-extract-dims ark,t:- ark:$tmpdir/$utt_id.mask.init.ark \
    ark:$tmpdir/$utt_id.vad.0.ark || exit 1
  
  gmm-global-get-frame-likes $init_speech_model \
    ark:$tmpdir/$utt_id.feats.init.ark ark:$tmpdir/$utt_id.speech_likes.init.ark || exit 1

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
    
    ### Compute likelihoods with the current Silence and Sound GMMs
    gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      ark:$tmpdir/$utt_id.feats.init.ark ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark || exit 1

    ### Get new VAD predictions on the subset selected for training
    ### Silence and Sound GMMs
    loglikes-to-class \
      ark:$tmpdir/$utt_id.silence_likes.$[x+1].ark \
      ark:$tmpdir/$utt_id.speech_likes.init.ark \
      ark:$tmpdir/$utt_id.vad.$[x+1].ark || exit 1
    
    loglikes-to-class \
      "ark:gmm-global-get-frame-likes $tmpdir/$utt_id.silence.$[x+1].mdl \
      \"$feats\" ark:- |" \
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
  loglikes-to-class \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$dir/$utt_id.speech_likes.bootstrap.ark \
    ark:$phase2_dir/$utt_id.seg.init.ark || exit 1

  ### Initialize Speech GMM
  select-top-chunks --window-size=1 \
    --selection-mask=ark:$phase2_dir/$utt_id.seg.init.ark --select-class=1 \
    "$feats" ark:- | gmm-global-init-from-feats \
    --num-gauss=$speech_num_gauss --num-iters=$[speech_num_gauss*2] \
    ark:- $phase2_dir/$utt_id.speech.0.mdl || exit 1
    
  gmm-global-get-frame-likes $phase2_dir/$utt_id.speech.0.mdl \
    "$feats" ark:$phase2_dir/$utt_id.speech_likes.init.ark || exit 1
  
  ### Compute initial segmentation for phase 2 training
  loglikes-to-class \
    ark:$phase2_dir/$utt_id.silence_likes.init.ark \
    ark:$phase2_dir/$utt_id.speech_likes.init.ark \
    ark:$phase2_dir/$utt_id.pred.init.ark || exit 1

  cp $tmpdir/$utt_id.silence.$x.mdl $phase2_dir/$utt_id.silence.0.mdl || exit 1
  {
    cat $dir/trans.mdl;
    echo "<DIMENSION> $feat_dim <NUMPDFS> 2";
    gmm-global-copy --binary=false $phase2_dir/$utt_id.silence.0.mdl -;
    gmm-global-copy --binary=false $phase2_dir/$utt_id.speech.0.mdl -;
  } | gmm-copy - $phase2_dir/$utt_id.0.mdl || exit 1

  x=0
  while [ $x -le $num_iters_phase2 ]; do
    gmm-latgen-faster --acoustic-scale=1.0 --determinize-lattice=false \
      $phase2_dir/$utt_id.$x.mdl $dir/graph/HCLG.fst \
      "$feats" "ark:| gzip -c > $phase2_dir/$utt_id.$x.lat.gz" \
      ark:/dev/null ark:- | \
      ali-to-phones --per-frame=true \
      $phase2_dir/$utt_id.$x.mdl ark:- ark:- | \
      copy-int-vector ark:- ark:$phase2_dir/$utt_id.$x.ali || exit 1
    
    lattice-to-post --acoustic-scale=1.0 \
      "ark:gunzip -c $phase2_dir/$utt_id.$x.lat.gz |" ark:- | \
      rand-prune-post 0.6 ark:- ark:- | \
      gmm-acc-stats $phase2_dir/$utt_id.$x.mdl "$feats" ark:- - | \
      gmm-est --mix-up=$[sil_num_gauss+speech_num_gauss] \
      --update-flags=tmv $phase2_dir/$utt_id.$x.mdl - $phase2_dir/$utt_id.$[x+1].mdl || exit 1


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

