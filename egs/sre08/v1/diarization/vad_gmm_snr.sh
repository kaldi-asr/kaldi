#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -u 
set -o pipefail

cmd=run.pl
stage=-100
try_merge_speech_noise=false
write_feats=false

## Features paramters
window_size=5             # 5 frame. Window over which initial selection of frames

. path.sh
. parse_options.sh || exit 1

if [ $# -ne 4 ]; then
  echo "Usage: vad_gmm_snr.sh <data> <init-silence-model> <init-speech-model> <dir>"
  echo " e.g.: vad_gmm_snr.sh data/rt05_eval exp/librispeech_s5/vad_model/{silence,speech}.0.mdl exp/vad_rt05_eval"
  exit 1
fi

data=$1
init_silence_model=$2
init_speech_model=$3
dir=$4

init_model_dir=`dirname $init_speech_model`
add_zero_crossing_feats=

# Prepare a lang directory
if [ $stage -le -4 ]; then
  mkdir -p $dir/local/dict
  mkdir -p $dir/local/lm
  mkdir -p $dir/local/dict_2class
  mkdir -p $dir/local/lm_2class

  echo "1" > $dir/local/dict/silence_phones.txt
  echo "1" > $dir/local/dict/optional_silence.txt
  echo "2" > $dir/local/dict/nonsilence_phones.txt
  echo "1" > $dir/local/dict_2class/silence_phones.txt
  echo "1" > $dir/local/dict_2class/optional_silence.txt
  echo "2" > $dir/local/dict_2class/nonsilence_phones.txt
  echo "3" >> $dir/local/dict/nonsilence_phones.txt
  echo -e "1 1\n2 2" > $dir/local/dict_2class/lexicon.txt
  echo -e "1 1\n2 2\n3 3" > $dir/local/dict/lexicon.txt
  echo -e "1\n2\n1 2" > $dir/local/dict_2class/extra_questions.txt
  echo -e "1\n2\n1 2\n3\n1 3\n2 3\n1 2 3" > $dir/local/dict/extra_questions.txt

  mkdir -p $dir/lang
  diarization/prepare_vad_lang.sh --num-sil-states $num_sil_states --num-nonsil-states $num_nonsil_states \
    $dir/local/dict $dir/local/lang $dir/lang || exit 1
  diarization/prepare_vad_lang.sh --num-sil-states $num_sil_states --num-nonsil-states $num_nonsil_states \
    $dir/local/dict_2class $dir/local/lang_2class $dir/lang_2class || exit 1
fi

feat_dim=`gmm-global-info $init_speech_model | grep "feature dimension" | awk '{print $NF}'` || exit 1

if [ $stage -le -3 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  
  run.pl $dir/log/create_transition_model_2class.log gmm-init-mono \
    $dir/lang_2class/topo $feat_dim - $dir/tree_2class \| \
    copy-transition-model --binary=false - $dir/trans_2class.mdl || exit 1

  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diarization/make_vad_graph.sh --iter trans_2class --tree tree_2class $dir/lang_2class $dir $dir/graph_2class || exit 1
fi

if [ $stage -le -2 ]; then
  {
    cat $dir/trans_2class.mdl
    echo "<DIMENSION> $feat_dim <NUMPDFS> 2"
    gmm-global-copy --binary=false $init_silence_model - || exit 1
    gmm-global-copy --binary=false $init_speech_model - || exit 1
  } | gmm-copy - $dir/init_2class.mdl || exit 1
fi

if [ $stage -le -1 ]; then
  t=$speech_to_sil_ratio
  lang=$dir/lang_test_${t}x
  cp -r $dir/lang $lang
  perl -e '$t = shift @ARGV; print "0 0 1 1 " . -log(1/($t+3)) . "\n0 0 2 2 ". -log($t/($t+3)). "\n0 0 3 3 ". -log(1/($t+3)) ."\n0 ". -log(1/($t+3))' $t | \
    fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
    > $lang/G.fst || exit 1
  diarization/make_vad_graph.sh --iter trans $lang $dir $dir/graph_test_${t}x || exit 1
  
  lang=$dir/lang_2class_test_${t}x
  cp -r $dir/lang_2class $lang
  perl -e '$t = shift @ARGV; print "0 0 1 1 " . -log(1/($t+2)) . "\n0 0 2 2 ". -log($t/($t+2)). "\n0 ". -log(1/($t+2))' $t | \
    fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
    > $lang/G.fst || exit 1
  
  diarization/make_vad_graph.sh --iter trans_2class --tree tree_2class $lang $dir $dir/graph_2class_test_${t}x || exit 1
fi

while IFS=$'\n' read line; do
  feats="ark:echo $line | apply-cmvn-sliding scp:- ark:- |${ignore_energy_opts}"

  utt_id=$(echo $line | awk '{print $1}')
  echo $utt_id > $dir/$utt_id.list

  if [ -f $data/segments ]; then
    if $add_zero_crossing_feats; then
      # Extract zero-crossing feats for adding as a feature
      $cmd $dir/log/$utt_id.extract_zero_crossings.log \
        utils/filter_scp.pl $dir/$utt_id.list $data/segments \| \
        extract-segments scp:$data/wav.scp - ark:- \| \
        compute-zero-crossings $zc_opts ark:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    fi
    
    # Extract log-energies
    $cmd $dir/log/$utt_id.extract_log_energies.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/segments \| \
      extract-segments scp:$data/wav.scp - ark:- \| \
      compute-mfcc-feats --config=conf/mfcc_vad.conf --num-ceps=1 \
      ark:- ark:- \| extract-column ark:- \
      ark:$dir/$utt_id.log_energies.ark || exit 1
  else 
    if $add_zero_crossing_feats; then
      # Extract zero-crossing feats for adding as a feature
      $cmd $dir/log/$utt_id.extract_zero_crossings.log \
        utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp \| \
        compute-zero-crossings $zc_opts scp:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    fi

    # Extract log-energies
    $cmd $dir/log/$utt_id.extract_log_energies.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp \| \
      compute-mfcc-feats --config=conf/mfcc_vad.conf --num-ceps=1 \
      scp:- ark:- \| extract-column ark:- \
      ark:$dir/$utt_id.log_energies.ark || exit 1
  fi
  
  [ -z "$frame_snrs_scp" ] && echo "$0: add-frame-snrs is true but frame-snrs-scp is not supplied" && exit 1
  utils/filter_scp.pl $data/utt2spk $frame_snrs_scp > $dir/frame_snrs.scp

  # Initial GMM parameters
  sil_num_gauss=$sil_num_gauss_init
  sound_num_gauss=$sound_num_gauss_init
  speech_num_gauss=$speech_num_gauss_init
  
  # Optionally add zero-crossings to the features
  if $add_zero_crossing_feats; then
    feats="${feats}paste-feats ark:- ark:$dir/$utt_id.zero_crossings.ark ark:- |" 
  fi

  # Optionally add frame-snrs to the features
  if $add_frame_snrs; then
    feats="${feats}paste-feats ark:- \"ark:vector-to-feat scp:$dir/frame_snrs.scp ark:- |\" ark:- |"
  fi
  
  # Add delta and delta-deltas
  feats="${feats} add-deltas ark:- ark:- |"

  if $write_feats; then
    copy-feats "$feats" ark:$dir/$utt_id.feat.ark
  fi
  
  # Compute initial likelihoods wrt to speech and silence models
  $cmd $dir/log/$utt_id.gmm_compute_likes.bootstrap.log \
    gmm-compute-likes $dir/init.mdl "$feats" \
    ark:$tmpdir/$utt_id.likes.bootstrap.ark &
  
  # Get VAD from bootstrap model. This is just for baseline.
  # This is not actually used later.
  $cmd $dir/log/$utt_id.get_vad.bootstrap.log \
    gmm-decode-simple --allow-partial=$allow_partial \
    --word-symbol-table=$dir/graph/words.txt \
    $dir/init.mdl $dir/graph/HCLG.fst \
    "$feats" ark:/dev/null ark:- \| ali-to-pdf $dir/init.mdl ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    ark:$tmpdir/$utt_id.vad.bootstrap.ark || exit 1
  
  # i.e. unless use-bootstrap-vad is given. (Only for baseline)
  if $use_bootstrap_vad; then
    segmentation-copy ark:$tmpdir/$utt_id.vad.bootstrap.ark \
      ark,scp:$dir/$utt_id.vad.final.ark,$dir/$utt_id.vad.final.scp || exit 1
    continue
  fi
  
  x=0
  goto_phase3=false   # Stage for merging speech and noise

  #############################################################################
  # Phase 1
  # Train noise GMM on lowest SNR frames. 
  # Train silence GMM on lowest energy frames.
  #############################################################################

  while [ $x -lt $num_iters ]; do
    # Number of frames to initially train the silence and sound GMMs
    num_frames_silence=$[num_frames_init_silence + sil_num_gauss * frames_per_gaussian ] 
    num_frames_sound=$[num_frames_init_sound + 5 * sound_num_gauss * frames_per_gaussian ]
    num_frames_sound_next=$[num_frames_init_sound_next + sound_num_gauss * frames_per_gaussian ]

    if [ $x -lt 3 ]; then
      $cmd $tmpdir/log/$utt_id.select_top.first.$[x+1].log \
        segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
        segmentation-select-top --num-bins=$num_bins \
        --merge-labels=0:2 --merge-dst-label=0 \
        --num-top-frames=$num_frames_sound --num-bottom-frames=$num_frames_silence \
        --top-select-label=2 --bottom-select-label=0 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$dir/$utt_id.log_energies.ark \
        ark:$tmpdir/$utt_id.seg.first.$[x+1].ark || exit 1

      
  done

