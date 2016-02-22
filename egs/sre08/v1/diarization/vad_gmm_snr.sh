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

if [ $# -ne 5 ]; then
  echo "Usage: vad_gmm_snr.sh <data> <frame-snrs-scp> <init-silence-model> <init-speech-model> <dir>"
  echo " e.g.: vad_gmm_snr.sh data/rt05_eval exp/librispeech_s5/vad_model/{silence,speech}.0.mdl exp/vad_rt05_eval"
  exit 1
fi

data=$1
frame_snrs_scp=$2
init_silence_model=$3
init_speech_model=$4
dir=$5

init_model_dir=`dirname $init_speech_model`
add_zero_crossing_feats=`cat $init_model_dir/add_zero_crossing_feats` || exit 1
add_frame_snrs=`cat $init_model_dir/add_frame_snrs` || exit 1

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
  
  cp $tmpdir/$utt_id.likes.bootstrap.ark $tmpdir/$utt_id.likes.0.ark
  
  x=0
  goto_phase3=false   # Stage for merging speech and noise

  #############################################################################
  # Phase 1
  # Train noise GMM on lowest SNR frames. 
  # Train speech GMM on highest likelihood frames
  # Train silence GMM on lowest energy frames.
  #############################################################################

  while [ $x -lt $num_iters ]; do
    # Number of frames to initially train the silence and sound GMMs
    num_frames_silence=$[num_frames_init_silence + sil_num_gauss * frames_per_gaussian ] 
    num_frames_silence_next=$[num_frames_init_silence_next + silence_num_gauss * frames_per_gaussian ]
    num_frames_sound=$[num_frames_init_sound + 5 * sound_num_gauss * frames_per_gaussian ]
    num_frames_sound_next=$[num_frames_init_sound_next + sound_num_gauss * frames_per_gaussian ]
    num_frames_speech=$[num_frames_init_speech + speech_num_gauss * frames_per_gaussian ]

    if [ $x -lt 3 ]; then
      # For the initial 3 iterations, the silence, sound and speech frames are 
      # defined as follows:
      # Silence -- low energy and low speech likelihood frames
      # Sound   -- low SNR and low speech likelihood frames
      # Speech  -- high speech likelihood

      # Find silence frames
      $cmd $tmpdir/log/$utt_id.select_top.silence.$x.log \
        segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
        segmentation-select-top --num-bins=$num_bins \
        --merge-labels=1 --merge-dst-label=0 \
        --num-bottom-frames=$num_frames_silence \
        --bottom-select-label=0 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$dir/$utt_id.log_energies.ark ark:- \| \
        segmentation-select-top --num-bins=$num_bins \
        --num-bottom-frames=$num_frames_silence_next \
        --bottom-select-label=0 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$tmpdir/$utt_id.likes.$x.ark \
        ark:$tmpdir/$utt_id.seg.silence.$x.ark || exit 1
      
      # Find noise frames
      $cmd $tmpdir/log/$utt_id.select_top.sound.$x.log \
        segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
        segmentation-select-top --num-bins=$num_bins \
        --merge-labels=1 --merge-dst-label=2 \
        --num-bottom-frames=$num_frames_sound \
        --bottom-select-label=2 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- "scp:utils/filter_scp.pl $dir/$utt_id.list $dir/frame_snrs.scp |" \| \
        segmentation-select-top --num-bins=$num_bins \
        --num-bottom-frames=$num_frames_sound_next \
        --bottom-select-label=2 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$tmpdir/$utt_id.likes.$x.ark \
        ark:$tmpdir/$utt_id.seg.sound.$x.ark || exit 1

      #$cmd $tmpdir/log/$utt_id.merge_segmentations.$x.log \
      #  segmentation-merge ark:$tmpdir/$utt_id.seg.silence.$[x+1].ark \
      #  ark:$tmpdir/$utt_id.seg.sound.$[x+1].ark \
      #  ark:$tmpdir/$utt_id.seg.$[x+1].ark

      # Find speech frames
      $cmd $tmpdir/log/$utt_id.select_top.speech.$x.log \
        segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
        segmentation-select-top --num-bins=$num_bins \
        --num-top-frames=$num_frames_speech \
        --top-select-label=1 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$tmpdir/$utt_id.likes.$x.ark \
        ark:$tmpdir/$utt_id.seg.speech.$x.ark || exit 1
    else
      # Get segmentation
      $cmd $tmpdir/log/$utt_id.get_seg.$x.log \
        gmm-decode-simple --allow-partial=$allow_partial \
        --word-symbol-table=$dir/graph/words.txt \
        $tmpdir/$utt_id.$x.mdl $dir/graph/HCLG.fst \
        "$feats" ark:/dev/null ark:- \| \
        ali-to-pdf $tmpdir/$utt_id.$x.mdl ark:- ark:- \| \
        segmentation-init-from-ali ark:- \
        ark:$tmpdir/$utt_id.seg.$x.ark || exit 1
    fi

    if [ $x -eq 0 ]; then
      {
        cat $dir/trans.mdl;
        echo "<DIMENSION> $feat_dim <NUMPDFS> 3";
        select-feats-from-segmentation --select-label=0 "$feats" \
          ark:$tmpdir/$utt_id.seg.silence.$x.ark ark:- | \
          gmm-global-init-from-feats --binary=false \
          --num-iters=$[sil_num_gauss+1] --num-gauss-init=1 --num-gauss=$sil_num_gauss \
          ark:- - || exit 1
        select-feats-from-segmentation --select-label=1 "$feats" \
          ark:$tmpdir/$utt_id.seg.speech.$x.ark ark:- | \
          gmm-global-init-from-feats --binary=false \
          --num-iters=$[speech_num_gauss+1] --num-gauss-init=1 --num-gauss=$speech_num_gauss \
          ark:- - || exit 1
        select-feats-from-segmentation --select-label=2 "$feats" \
          ark:$tmpdir/$utt_id.seg.sound.$x.ark ark:- | \
          gmm-global-init-from-feats --binary=false \
          --num-iters=$[sound_num_gauss+1] --num-gauss-init=1 --num-gauss=$sound_num_gauss \
          ark:- - || exit 1
      } 2> $tmpdir/log/$utt_id.init_gmm.log | \
        gmm-copy - $tmpdir/$utt_id.$[x+1].mdl 2>> $tmpdir/log/$utt_id.init_gmm.log
      if [ $? -ne 0 ]; then
        echo "Insufficient frames for training silence or sound model. Skipping to phase 3"
        goto_phase3=true
        break;
      fi
    else
      $cmd $tmpdir/log/$utt_id.gmm_update.$[x+1].log \
        gmm-update-segmentation \
        --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n1 $speech_num_gauss\n2 $sound_num_gauss\" |" \
        $tmpdir/$utt_id.$x.mdl "$feats" \
        ark:$tmpdir/$utt_id.seg.$x.ark \
        $tmpdir/$utt_id.$[x+1].mdl || exit 1
    fi

    $cmd $tmpdir/log/$utt_id.gmm_compute_likes.$[x+1].log \
      gmm-compute-likes $tmpdir/$utt_id.$[x+1].mdl "$feats" \
      ark:$tmpdir/$utt_id.likes.$[x+1].ark &
    
    if [ $sil_num_gauss -lt $sil_max_gauss ]; then
      sil_num_gauss=$[sil_num_gauss + sil_gauss_incr]
    fi

    if [ $sound_num_gauss -lt $sound_max_gauss ]; then
      sound_num_gauss=$[sound_num_gauss + sound_gauss_incr]
    fi

    if [ $speech_num_gauss -lt $speech_max_gauss ]; then
      speech_num_gauss=$[speech_num_gauss + speech_gauss_incr]
    fi

    x=$[x+1]
  done    ## Done training GMMs
  echo "$0: Phase 1 training done!"
    
  cp $tmpdir/$utt_id.$x.mdl $dir/$utt_id.final.mdl
  rm -f $dir/$utt_id.graph_final
  ln -s graph_test_${speech_to_sil_ratio}x $dir/$utt_id.graph_final

  # Get final segmentation at the end of phase 1
  $cmd $tmpdir/log/$utt_id.get_seg.$x.log \
    gmm-decode-simple --allow-partial=$allow_partial \
    --word-symbol-table=$dir/graph/words.txt \
    $tmpdir/$utt_id.$x.mdl $dir/graph/HCLG.fst \
    "$feats" ark:/dev/null ark:- \| \
    ali-to-pdf $tmpdir/$utt_id.$x.mdl ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    ark:$tmpdir/$utt_id.seg.$x.ark || exit 1
    
  mkdir -p $phase3_dir/log

  #############################################################################
  # Try merging speech and noise GMMs
  #############################################################################

  # Create a merged model
  $cmd $tmpdir/log/$utt_id.init_nonsil.log \
    segmentation-copy --merge-labels=1:2 --merge-dst-label=1 \
    ark:$tmpdir/$utt_id.seg.$x.ark ark:- \| \
    select-feats-from-segmentation --select-label=1 \
      "$feats" ark:- ark:- \| \
      gmm-global-init-from-feats \
      --num-iters=$[sound_num_gauss + speech_num_gauss + 1] \
      --num-gauss-init=1 \
      --num-gauss=$[sound_num_gauss + speech_num_gauss] ark:- \
      $tmpdir/$utt_id.$x.nonsil.mdl || exit 1
    
  # Select speech feats from the final segmentation
  $cmd $tmpdir/log/$utt_id.select_speech_feats.$x.log \
    select-feats-from-segmentation --select-label=1 \
      "$feats" ark:$tmpdir/$utt_id.seg.$x.ark \
      ark:$tmpdir/$utt_id.speech_feats.$x.ark

  if [ $? -eq 0 ]; then
    # Check if there is sufficient speech frames
    num_selected_speech=$(grep "Processed .* segmentations; selected" $tmpdir/log/$utt_id.select_speech_feats.$x.log | perl -pe 's/.+selected (\S+) out of \S+ frames/$1/')
    if [ $num_selected_speech -lt $min_data ]; then
      echo "Insufficient frames for speech at the end of phase 1. $num_selected_speech < $min_data. See $tmpdir/log/$utt_id.select_speech_feats.$x.log. Going to phase 3."
      goto_phase3=true
    fi
  else
    # Check if there is any speech frame
    echo "Failed to find any data for speech at the end of phase 1. See $tmpdir/log/$utt_id.select_speech_feats.$x.log. Going to phase 3."
    goto_phase3=true
  fi

  if ! $goto_phase3; then
    # Not failed yet. So can compute speech likelihood.
    speech_like=$(gmm-global-get-frame-likes \
      "gmm-extract-pdf $tmpdir/$utt_id.$x.mdl 1 - |" \
      ark:$tmpdir/$utt_id.speech_feats.$x.ark ark,t:- | \
      perl -pe 's/.*\[(.+)]/$1/' | \
      perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $tmpdir/$utt_id.compute_speech_like.$x.log  || exit 1

    # Select noise feats from the final segmentation
    $cmd $tmpdir/log/$utt_id.select_sound_feats.$x.log \
      select-feats-from-segmentation --select-label=2 \
        "$feats" ark:$tmpdir/$utt_id.seg.$x.ark \
        ark:$tmpdir/$utt_id.sound_feats.$x.ark

    if [ $? -eq 0 ]; then
      # Check if there is sufficient noise frames
      num_selected_sound=$(grep "Processed .* segmentations; selected" $tmpdir/log/$utt_id.select_sound_feats.$x.log | perl -pe 's/.+selected (\S+) out of \S+ frames/$1/')
      if [ $num_selected_sound -lt $min_data ]; then
        echo "Insufficient frames for sound at the end of phase 1. $num_selected_sound < $min_data. See $tmpdir/log/$utt_id.select_sound_feats.$x.log. Going to phase 3."
        goto_phase3=true
      fi
    else
      # Check if there is any noise frame
      echo "Failed to find any data for sound at the end of phase 1. See $phase2_dir/log/$utt_id.select_sound_feats.$x.log. Going to phase 3."
      goto_phase3=true
    fi
  fi

  if ! $goto_phase3; then
    # Not failed yet. So can compute noise likelihood.
    sound_like=$(gmm-global-get-frame-likes \
      "gmm-extract-pdf $phase2_dir/$utt_id.$x.mdl 2 - |" \
      ark:$phase2_dir/$utt_id.sound_feats.$x.ark ark,t:- | \
      perl -pe 's/.*\[(.+)]/$1/' | \
      perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $phase2_dir/$utt_id.compute_sound_like.$x.log  || exit 1

    # Compute non-silence likelihood using combined speech+noise GMM
    nonsil_like=$(select-feats-from-segmentation --merge-labels=1:2 --select-label=1 \
      "$feats" ark:$phase2_dir/$utt_id.seg.$x.ark ark:- | \
      gmm-global-get-frame-likes \
      $phase2_dir/$utt_id.$x.nonsil.mdl ark:- ark,t:- | \
      perl -pe 's/.*\[(.+)]/$1/' | \
      perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $phase2_dir/$utt_id.compute_nonsil_like.$x.log  || exit 1

    # Likelihood test -- Check if the combined model gives better likelihood 
    # than separate speech and noise models. If yes, then go to phase 3
    if [ ! -z `perl -e "print \"true\" if ($sound_like + $speech_like < $nonsil_like)"` ]; then
      goto_phase3=true
    fi
  fi

  if $goto_phase3; then
    #############################################################################
    # Phase 3
    # Train speech GMM on highest likelihood frames
    # Train silence GMM on lowest energy frames.
    #############################################################################

    while [ $x -lt $num_iters ]; do
      # Number of frames to initially train the silence and sound GMMs
      num_frames_silence=$[num_frames_init_silence + sil_num_gauss * frames_per_gaussian ] 
      num_frames_silence_next=$[num_frames_init_silence_next + silence_num_gauss * frames_per_gaussian ]
      num_frames_speech=$[num_frames_init_speech + speech_num_gauss * frames_per_gaussian ]

      if [ $x -lt 3 ]; then
        # For the initial 3 iterations, the silence and speech frames are 
        # defined as follows:
        # Silence -- low energy and low speech likelihood frames
        # Speech  -- high speech likelihood

        # Find silence frames
        $cmd $phase3_dir/log/$utt_id.select_top.silence.$x.log \
          segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
          segmentation-select-top --num-bins=$num_bins \
          --merge-labels=1 --merge-dst-label=0 \
          --num-bottom-frames=$num_frames_silence \
          --bottom-select-label=0 --reject-label=1000 \
          --remove-rejected-frames=true \
          --window-size=$window_size --min-window-remainder=$[window_size/2] \
          ark:- ark:$dir/$utt_id.log_energies.ark ark:- \| \
          segmentation-select-top --num-bins=$num_bins \
          --num-bottom-frames=$num_frames_silence_next \
          --bottom-select-label=0 --reject-label=1000 \
          --remove-rejected-frames=true \
          --window-size=$window_size --min-window-remainder=$[window_size/2] \
          ark:- ark:$phase3_dir/$utt_id.likes.$x.ark \
          ark:$phase3_dir/$utt_id.seg.silence.$x.ark || exit 1

        # Find speech frames
        $cmd $phase3_dir/log/$utt_id.select_top.speech.$x.log \
          segmentation-init-from-lengths "ark:echo $line | feat-to-len scp:- ark:- |" | \
          segmentation-select-top --num-bins=$num_bins \
          --num-top-frames=$num_frames_speech \
          --top-select-label=1 --reject-label=1000 \
          --remove-rejected-frames=true \
          --window-size=$window_size --min-window-remainder=$[window_size/2] \
          ark:- ark:$phase3_dir/$utt_id.likes.$x.ark ark:- \| \
          segmentation-select-top --num-bins=$num_bins \
          --num-top-frames=$num_frames_speech \
          ark:$phase3_dir/$utt_id.seg.speech.$x.ark || exit 1
      else
        # Get segmentation using current model
        $cmd $phase3_dir/log/$utt_id.get_seg.$x.log \
          gmm-decode-simple --allow-partial=$allow_partial \
          --word-symbol-table=$dir/graph_2class/words.txt \
          $phase3_dir/$utt_id.$[x+1].mdl $dir/graph/HCLG.fst \
          "$feats" ark:/dev/null ark:- \| \
          ali-to-pdf $phase3_dir/$utt_id.$x.mdl ark:- ark:- \| \
          segmentation-init-from-ali ark:- \
          ark:$phase3_dir/$utt_id.seg.$x.ark || exit 1
      fi

      if [ $x -eq 0 ]; then
        {
          cat $dir/trans_2class.mdl;
          echo "<DIMENSION> $feat_dim <NUMPDFS> 3";
          select-feats-from-segmentation --select-label=0 "$feats" \
            ark:$phase3_dir/$utt_id.seg.silence.$x.ark ark:- | \
            gmm-global-init-from-feats --binary=false \
            --num-iters=$[sil_num_gauss+1] --num-gauss-init=1 --num-gauss=$sil_num_gauss \
            ark:- - || exit 1
          select-feats-from-segmentation --select-label=1 "$feats" \
            ark:$phase3_dir/$utt_id.seg.speech.$x.ark ark:- | \
            gmm-global-init-from-feats --binary=false \
            --num-iters=$[speech_num_gauss+1] --num-gauss-init=1 --num-gauss=$speech_num_gauss \
            ark:- - || exit 1
        } 2> $phase3_dir/log/$utt_id.init_gmm.log | \
          gmm-copy - $phase3_dir/$utt_id.$[x+1].mdl 2>> $phase3_dir/log/$utt_id.init_gmm.log
        if [ $? -ne 0 ]; then
          echo "VAD failed for utterance $utt_id. Utterance is fully silence." 
          fail_flag=true
          break
        fi
      else
        $cmd $phase3_dir/log/$utt_id.gmm_update.$[x+1].log \
          gmm-update-segmentation \
          --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n1 $speech_num_gauss\n2 $sound_num_gauss\" |" \
          $phase3_dir/$utt_id.$x.mdl "$feats" \
          ark:$phase3_dir/$utt_id.seg.$x.ark \
          $phase3_dir/$utt_id.$[x+1].mdl || exit 1
      fi

      $cmd $phase3_dir/log/$utt_id.gmm_compute_likes.$[x+1].log \
        gmm-compute-likes $phase3_dir/$utt_id.$[x+1].mdl "$feats" \
        ark:$phase3_dir/$utt_id.likes.$[x+1].ark &

      if [ $sil_num_gauss -lt $sil_max_gauss ]; then
        sil_num_gauss=$[sil_num_gauss + sil_gauss_incr]
      fi

      if [ $speech_num_gauss -lt $speech_max_gauss ]; then
        speech_num_gauss=$[speech_num_gauss + speech_gauss_incr]
      fi

      x=$[x+1]
    done    ## Done training GMMs
    echo "$0: Phase 3 training done!"

    cp $phase3_dir/$utt_id.$x.mdl $dir/$utt_id.final.mdl
    rm -f $dir/$utt_id.graph_final
    ln -s graph_2class_test_${speech_to_sil_ratio}x $dir/$utt_id.graph_final
  fi

  if ! $fail_flag; then
    $cmd $dir/log/$utt_id.gmm_compute_likes.final.log \
      gmm-compute-likes $dir/$utt_id.final.mdl "$feats" \
      ark:$dir/$utt_id.likes.final.ark &

    $cmd $dir/log/$utt_id.get_seg.final.log \
      gmm-decode-simple --allow-partial=$allow_partial \
      --word-symbol-table=$dir/$utt_id.graph_final/words.txt \
      $dir/$utt_id.final.mdl $dir/$utt_id.graph_final/HCLG.fst \
      "$feats" ark:/dev/null ark:- \| \
      ali-to-pdf $dir/$utt_id.final.mdl ark:- ark:- \| \
      segmentation-init-from-ali ark:- \
      ark,scp:$dir/$utt_id.vad.final.ark,$dir/$utt_id.vad.final.scp || exit 1
  fi

done < $data/feats.scp
