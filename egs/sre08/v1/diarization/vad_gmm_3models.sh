#!/bin/bash
# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -u 
set -o pipefail

cmd=run.pl
stage=-1
allow_partial=true
try_merge_speech_noise=false

## Features paramters
window_size=100                   # 1s
min_data=200
frames_per_gaussian=2000
num_bins=100
num_sil_states=30
num_nonsil_states=75

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
sample_per_gaussian=2000
num_iters_init=3
num_iters=5
min_sil_variance=1
min_sound_variance=0.01
min_speech_variance=0.001

## Phase 2 parameters
num_frames_init_speech=100000
speech_num_gauss_init=6
sil_max_gauss_phase2=7
sound_max_gauss_phase2=18
speech_max_gauss_phase2=16
sil_gauss_incr_phase2=1
sound_gauss_incr_phase2=2
speech_gauss_incr_phase2=2
num_iters_phase2=5
window_size_phase2=10

## Phase 3 parameters
num_frames_silence_phase3_init=2000
num_frames_speech_phase3_init=2000
sil_num_gauss_init_phase3=2
speech_num_gauss_init_phase3=2
sil_max_gauss_phase3=5
sil_max_gauss_phase4=8
speech_max_gauss_phase4=16
sil_gauss_incr_phase3=1
sil_gauss_incr_phase4=1
speech_gauss_incr_phase4=2
num_iters_phase3=5
num_iters_phase4=5

speech_to_sil_ratio=1

. path.sh
. parse_options.sh || exit 1

if [ $# -ne 5 ]; then
  echo "Usage: vad_gmm_icsi.sh <data> <init-silence-model> <init-speech-model> <init-noise-model> <dir>"
  echo " e.g.: vad_gmm_icsi.sh data/rt05_eval exp/librispeech_s5/vad_model/{silence,speech,noise}.0.mdl exp/vad_rt05_eval"
  exit 1
fi

data=$1
init_silence_model=$2
init_speech_model=$3
init_sound_model=$4
dir=$5

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

# Prepare a lang directory
if [ $stage -le -12 ]; then
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

if [ $stage -le -11 ]; then 
  run.pl $dir/log/create_transition_model.log gmm-init-mono \
    $dir/lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
  
  run.pl $dir/log/create_transition_model_2class.log gmm-init-mono \
    $dir/lang_2class/topo $feat_dim - $dir/tree_2class \| \
    copy-transition-model --binary=false - $dir/trans_2class.mdl || exit 1

  diarization/make_vad_graph.sh --iter trans $dir/lang $dir $dir/graph || exit 1
  diarization/make_vad_graph.sh --iter trans_2class --tree tree_2class $dir/lang_2class $dir $dir/graph_2class || exit 1
fi

if [ $stage -le -10 ]; then
  {
    cat $dir/trans.mdl
    echo "<DIMENSION> $feat_dim <NUMPDFS> 3"
    gmm-global-copy --binary=false $init_silence_model - || exit 1
    gmm-global-copy --binary=false $init_speech_model - || exit 1
    gmm-global-copy --binary=false $init_sound_model - || exit 1
  } | gmm-copy - $dir/init.mdl || exit 1
fi

if [ $stage -le -9 ]; then
  t=$speech_to_sil_ratio
  lang=$dir/lang_test_${t}x
  cp -r $dir/lang $lang
  perl -e "print \"0 0 1 1 \" . -log(1/$[t+3]) . \"\n0 0 2 2 \". -log($t/$[t+3]). \"\n0 0 3 3 \". -log(1/$[t+3]) .\"\n0 \". -log(1/$[t+3])" | \
    fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
    > $lang/G.fst || exit 1
  diarization/make_vad_graph.sh --iter trans $lang $dir $dir/graph_test_${t}x || exit 1
  
  lang=$dir/lang_2class_test_${t}x
  cp -r $dir/lang_2class $lang
  perl -e "print \"0 0 1 1 \" . -log(1/$[t+2]) . \"\n0 0 2 2 \". -log($t/$[t+2]). \"\n0 \". -log(1/$[t+2])" | \
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
    $cmd $dir/log/$utt_id.extract_zero_crossings.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/segments \| \
      extract-segments scp:$data/wav.scp - ark:- \| \
      compute-zero-crossings $zc_opts ark:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    #$cmd $dir/log/$utt_id.extract_pitch.log \
    #  utils/filter_scp.pl $dir/$utt_id.list $data/segments \| \
    #  extract-segments scp:$data/wav.scp - ark:- \| \
    #  compute-kaldi-pitch-feats --config=conf/pitch.conf --frames-per-chunk=10 --simulate-first-pass-online=true \
    #  ark:- ark:$dir/$utt_id.kaldi_pitch.ark || exit 1
    $cmd $dir/log/$utt_id.extract_log_energies.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/segments \| \
      extract-segments scp:$data/wav.scp - ark:- \| \
      compute-mfcc-feats --config=conf/mfcc_vad.conf --num-ceps=1 \
      ark:- ark:- \| extract-column ark:- \
      ark:$dir/$utt_id.log_energies.ark || exit 1
  else 
    $cmd $dir/log/$utt_id.extract_zero_crossings.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp \| \
      compute-zero-crossings $zc_opts scp:- ark:$dir/$utt_id.zero_crossings.ark || exit 1
    #$cmd $dir/log/$utt_id.extract_pitch.log \
    #  utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp \| \
    #  compute-kaldi-pitch-feats --config=conf/pitch.conf --frames-per-chunk=10 --simulate-first-pass-online=true \
    #  scp:- ark:$dir/$utt_id.kaldi_pitch.ark || exit 1
    $cmd $dir/log/$utt_id.extract_log_energies.log \
      utils/filter_scp.pl $dir/$utt_id.list $data/wav.scp \| \
      compute-mfcc-feats --config=conf/mfcc_vad.conf --num-ceps=1 \
      scp:- ark:- \| extract-column ark:- \
      ark:$dir/$utt_id.log_energies.ark || exit 1
  fi

  sil_num_gauss=$sil_num_gauss_init
  sound_num_gauss=$sound_num_gauss_init
  speech_num_gauss=$speech_num_gauss_init
  
  if $add_zero_crossing_feats; then
    feats="${feats} paste-feats ark:- ark:$dir/$utt_id.zero_crossings.ark ark:- |" 
  fi

  feats="${feats} add-deltas ark:- ark:- |"

  # Get VAD: 0 for silence, 1 for speech and 2 for sound
  $cmd $dir/log/$utt_id.get_vad.bootstrap.log \
    gmm-decode-simple --allow-partial=$allow_partial \
    --word-symbol-table=$dir/graph/words.txt \
    $dir/init.mdl $dir/graph/HCLG.fst \
    "$feats" ark:/dev/null ark:- \| ali-to-pdf $dir/init.mdl ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    ark:$tmpdir/$utt_id.vad.bootstrap.ark || exit 1
 
  cp $tmpdir/$utt_id.vad.bootstrap.ark $tmpdir/$utt_id.seg.0.ark 

  x=0
  goto_phase3=false

  while [ $x -lt $num_iters ]; do
    num_frames_silence=$[num_frames_init_silence + sil_num_gauss * frames_per_gaussian ] 
    num_frames_sound=$[num_frames_init_sound + 5 * sound_num_gauss * frames_per_gaussian ]
    num_frames_sound_next=$[num_frames_init_sound_next + sound_num_gauss * frames_per_gaussian ]
    
    if [ $x -lt 3 ]; then
      $cmd $tmpdir/log/$utt_id.select_top.first.$[x+1].log \
        segmentation-copy --filter-label=0 \
        --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
        ark:$tmpdir/$utt_id.seg.$x.ark ark:- \| \
        segmentation-select-top --num-bins=$num_bins \
        --merge-labels=0:2 --merge-dst-label=0 \
        --num-top-frames=$num_frames_sound --num-bottom-frames=$num_frames_silence \
        --top-select-label=2 --bottom-select-label=0 --reject-label=1000 \
        --remove-rejected-frames=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:- ark:$dir/$utt_id.log_energies.ark \
        ark:$tmpdir/$utt_id.seg.first.$[x+1].ark || exit 1

      $cmd $tmpdir/log/$utt_id.select_top.$[x+1].log \
        segmentation-select-top --num-bins=$num_bins --src-label=2 \
        --num-top-frames=$num_frames_sound_next --num-bottom-frames=-1 \
        --top-select-label=2 --bottom-select-label=-1 --reject-label=1001 \
        --remove-rejected-frames=true --select-from-full-histogram=true \
        --window-size=$window_size --min-window-remainder=$[window_size/2] \
        ark:$tmpdir/$utt_id.seg.first.$[x+1].ark "ark:extract-column ark:$dir/$utt_id.zero_crossings.ark ark:- |" \
        ark:$tmpdir/$utt_id.seg.second.$[x+1].ark || exit 1
    else
      $cmd $tmpdir/log/$utt_id.select_top.$[x+1].log \
        segmentation-copy --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
        --filter-label=0 ark:$tmpdir/$utt_id.seg.$x.ark \
        ark:$tmpdir/$utt_id.seg.second.$[x+1].ark || exit 1
    fi

    if [ $x -eq 0 ]; then
      {
        cat $dir/trans.mdl;
        echo "<DIMENSION> $feat_dim <NUMPDFS> 3";
        select-feats-from-segmentation --select-label=0 "$feats" \
          ark:$tmpdir/$utt_id.seg.second.$[x+1].ark ark:- | \
          gmm-global-init-from-feats --binary=false \
          --num-iters=$[sil_num_gauss+1] --num-gauss-init=1 --num-gauss=$sil_num_gauss \
          ark:- - || exit 1
        gmm-global-copy --binary=false $init_speech_model -;
        select-feats-from-segmentation --select-label=2 "$feats" \
          ark:$tmpdir/$utt_id.seg.second.$[x+1].ark ark:- | \
          gmm-global-init-from-feats --binary=false \
          --num-iters=$[sound_num_gauss+1] --num-gauss-init=1 --num-gauss=$sound_num_gauss \
          ark:- - || exit 1
      } 2> $tmpdir/log/$utt_id.init_gmm.log | \
        gmm-copy - $tmpdir/$utt_id.$[x+1].mdl 2>> $tmpdir/log/$utt_id.init_gmm.log || exit 1
    else
      #$cmd $tmpdir/log/$utt_id.gmm_update.$[x+1].log \
      #  gmm-est-segmentation --pdfs=0:2 \
      #  --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n2 $sound_num_gauss\" |" \
      #  $tmpdir/$utt_id.$x.mdl "$feats" \
      #  ark:$tmpdir/$utt_id.seg.second.$[x+1].ark \
      #  $tmpdir/$utt_id.$[x+1].mdl || exit 1
      $cmd $tmpdir/log/$utt_id.gmm_update.$[x+1].log \
        gmm-update-segmentation --pdfs=0:2 \
        --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n2 $sound_num_gauss\" |" \
        $tmpdir/$utt_id.$x.mdl "$feats" \
        ark:$tmpdir/$utt_id.seg.second.$[x+1].ark \
        $tmpdir/$utt_id.$[x+1].mdl || exit 1
    fi
    
    $cmd $tmpdir/log/$utt_id.get_seg.$[x+1].log \
      gmm-decode-simple --allow-partial=$allow_partial \
      --word-symbol-table=$dir/graph/words.txt \
      $tmpdir/$utt_id.$[x+1].mdl $dir/graph/HCLG.fst \
      "$feats" ark:/dev/null ark:- \| \
      ali-to-pdf $tmpdir/$utt_id.$[x+1].mdl ark:- ark:- \| \
      segmentation-init-from-ali ark:- \
      ark:$tmpdir/$utt_id.seg.$[x+1].ark || exit 1

    if [ $sil_num_gauss -lt $sil_max_gauss ]; then
      sil_num_gauss=$[sil_num_gauss + sil_gauss_incr]
    fi

    if [ $sound_num_gauss -lt $sound_max_gauss ]; then
      sound_num_gauss=$[sound_num_gauss + sound_gauss_incr]
    fi

    x=$[x+1]
  done    ## Done training Silence and Speech GMMs
  
  $cmd $phase2_dir/log/$utt_id.init_speech.log \
    segmentation-copy --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
    --filter-label=1 ark:$tmpdir/$utt_id.seg.$num_iters.ark ark:- \| \
    select-feats-from-segmentation --select-label=1 "$feats" \
      ark:- ark:- \| \
      gmm-global-init-from-feats \
      --num-iters=$[speech_num_gauss+1] --num-gauss-init=1 --num-gauss=$speech_num_gauss \
      ark:- $phase2_dir/$utt_id.speech.0.mdl
  if [ $? -eq 0 ]; then
    num_selected_speech=$(grep "Processed .* segmentations; selected" $phase2_dir/log/$utt_id.init_speech.log | perl -pe 's/.+selected (\S+) out of \S+ frames/$1/')
    if [ $num_selected_speech -lt $min_data ]; then
      echo "Insufficient frames for speech at the end of phase 1. $num_selected_speech < $min_data. See $phase2_dir/log/$utt_id.init_speech.log. Going to phase 3."
      goto_phase3=true
    fi
  else
    echo "Failed to find any data for speech at the end of phase 1. See $phase2_dir/log/$utt_id.init_speech.log. Going to phase 3."
    goto_phase3=true
  fi

  if $goto_phase3; then
    rm -f $dir/$utt_id.current_seg.ark
    ln -s $tmpdir/$utt_id.seg.$x.ark $dir/$utt_id.current_seg.ark
  fi

  if ! $goto_phase3; then
    $cmd $phase2_dir/log/$utt_id.init_gmm.log \
      gmm-init-pdf-from-global $tmpdir/$utt_id.$num_iters.mdl 1 \
      $phase2_dir/$utt_id.speech.0.mdl $phase2_dir/$utt_id.0.mdl || exit 1

    x=0
    while [ $x -lt $num_iters_phase2 ]; do
      if [ $sil_num_gauss -lt $sil_max_gauss_phase2 ]; then
        sil_num_gauss=$[sil_num_gauss + sil_gauss_incr_phase2]
      fi

      if [ $sound_num_gauss -lt $sound_max_gauss_phase2 ]; then
        sound_num_gauss=$[sound_num_gauss + sound_gauss_incr_phase2]
      fi

      if [ $speech_num_gauss -lt $speech_max_gauss_phase2 ]; then
        speech_num_gauss=$[speech_num_gauss + speech_gauss_incr_phase2]
      fi

      $cmd $phase2_dir/log/$utt_id.get_seg.$x.log \
        gmm-decode-simple --allow-partial=$allow_partial \
        --word-symbol-table=$dir/graph/words.txt \
        $phase2_dir/$utt_id.$x.mdl $dir/graph/HCLG.fst \
        "$feats" ark:/dev/null ark:- \| \
        ali-to-pdf $phase2_dir/$utt_id.$x.mdl ark:- ark:- \| \
        segmentation-init-from-ali ark:- \
        ark:$phase2_dir/$utt_id.seg.$x.ark || exit 1
      
      #$cmd $phase2_dir/log/$utt_id.gmm_update.$[x+1].log \
      #  gmm-est-segmentation \
      #  --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n1 $speech_num_gauss\n2 $sound_num_gauss\" |" \
      #  $phase2_dir/$utt_id.$x.mdl "$feats" \
      #  ark:$phase2_dir/$utt_id.seg.$x.ark \
      #  $phase2_dir/$utt_id.$[x+1].mdl || exit 1
      $cmd $phase2_dir/log/$utt_id.gmm_update.$[x+1].log \
        gmm-update-segmentation \
        --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n1 $speech_num_gauss\n2 $sound_num_gauss\" |" \
        $phase2_dir/$utt_id.$x.mdl "$feats" \
        ark:$phase2_dir/$utt_id.seg.$x.ark \
        $phase2_dir/$utt_id.$[x+1].mdl || exit 1
      
      x=$[x+1]
    done  ## Done training all 3 GMMs
    cp $phase2_dir/$utt_id.$x.mdl $dir/$utt_id.final.mdl
    rm -f $dir/$utt_id.graph_final
    ln -s graph_test_${speech_to_sil_ratio}x $dir/$utt_id.graph_final

    $cmd $phase2_dir/log/$utt_id.get_seg.$x.log \
      gmm-decode-simple --allow-partial=$allow_partial \
      --word-symbol-table=$dir/graph/words.txt \
      $phase2_dir/$utt_id.$x.mdl $dir/graph/HCLG.fst \
      "$feats" ark:/dev/null ark:- \| \
      ali-to-pdf $phase2_dir/$utt_id.$x.mdl ark:- ark:- \| \
      segmentation-init-from-ali ark:- \
      ark:$phase2_dir/$utt_id.seg.$x.ark || exit 1

    mkdir -p $phase3_dir/log
    
    $cmd $phase2_dir/log/$utt_id.init_nonsil.log \
      segmentation-copy --merge-labels=1:2 --merge-dst-label=1 \
      ark:$phase2_dir/$utt_id.seg.$x.ark ark:- \| \
      select-feats-from-segmentation --select-label=1 \
        "$feats" ark:- ark:- \| \
        gmm-global-init-from-feats \
        --num-iters=$[sound_num_gauss + speech_num_gauss + 1] \
        --num-gauss-init=1 \
        --num-gauss=$[sound_num_gauss + speech_num_gauss] ark:- \
        $phase2_dir/$utt_id.$x.nonsil.mdl || exit 1
  
    $cmd $phase2_dir/log/$utt_id.select_speech_feats.$x.log \
      select-feats-from-segmentation --select-label=1 \
      "$feats" ark:$phase2_dir/$utt_id.seg.$x.ark \
      ark:$phase2_dir/$utt_id.speech_feats.$x.ark
    
    if $goto_phase3; then
      rm -f $dir/$utt_id.current_seg.ark
      ln -s $phase2_dir/$utt_id.seg.$x.ark $dir/$utt_id.current_seg.ark
    fi

    if [ $? -eq 0 ]; then
      num_selected_speech=$(grep "Processed .* segmentations; selected" $phase2_dir/log/$utt_id.select_speech_feats.$x.log | perl -pe 's/.+selected (\S+) out of \S+ frames/$1/')
      if [ $num_selected_speech -lt $min_data ]; then
        echo "Insufficient frames for speech at the end of phase 2. $num_selected_speech < $min_data. See $phase2_dir/log/$utt_id.select_speech_feats.$x.log. Going to phase 3."
        goto_phase3=true
      fi
    else
      echo "Failed to find any data for speech at the end of phase 1. See $phase2_dir/log/$utt_id.select_speech_feats.$x.log. Going to phase 3."
      goto_phase3=true
    fi

    if $try_merge_speech_noise; then
      if ! $goto_phase3; then
        speech_like=$(gmm-global-get-frame-likes \
          "gmm-extract-pdf $phase2_dir/$utt_id.$x.mdl 1 - |" \
          ark:$phase2_dir/$utt_id.speech_feats.$x.ark ark,t:- | \
          perl -pe 's/.*\[(.+)]/$1/' | \
          perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $phase2_dir/$utt_id.compute_speech_like.$x.log  || exit 1

        $cmd $phase2_dir/log/$utt_id.select_sound_feats.$x.log \
          select-feats-from-segmentation --select-label=2 \
            "$feats" ark:$phase2_dir/$utt_id.seg.$x.ark \
            ark:$phase2_dir/$utt_id.sound_feats.$x.ark

        if [ $? -eq 0 ]; then
          num_selected_sound=$(grep "Processed .* segmentations; selected" $phase2_dir/log/$utt_id.select_sound_feats.$x.log | perl -pe 's/.+selected (\S+) out of \S+ frames/$1/')
          if [ $num_selected_sound -lt $min_data ]; then
            echo "Insufficient frames for sound at the end of phase 2. $num_selected_sound < $min_data. See $phase2_dir/log/$utt_id.select_sound_feats.$x.log. Going to phase 3."
            goto_phase3=true
          fi
        else
          echo "Failed to find any data for sound at the end of phase 1. See $phase2_dir/log/$utt_id.select_sound_feats.$x.log. Going to phase 3."
          goto_phase3=true
        fi
      fi

      if ! $goto_phase3; then
        sound_like=$(gmm-global-get-frame-likes \
          "gmm-extract-pdf $phase2_dir/$utt_id.$x.mdl 2 - |" \
          ark:$phase2_dir/$utt_id.sound_feats.$x.ark ark,t:- | \
          perl -pe 's/.*\[(.+)]/$1/' | \
          perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $phase2_dir/$utt_id.compute_sound_like.$x.log  || exit 1

        nonsil_like=$(select-feats-from-segmentation --merge-labels=1:2 --select-label=1 \
          "$feats" ark:$phase2_dir/$utt_id.seg.$x.ark ark:- | \
          gmm-global-get-frame-likes \
          $phase2_dir/$utt_id.$x.nonsil.mdl ark:- ark,t:- | \
          perl -pe 's/.*\[(.+)]/$1/' | \
          perl -ane '$sum = 0; foreach(@F) { $sum = $sum + $_; $i = $i + 1;}; print STDOUT ($sum)') 2> $phase2_dir/$utt_id.compute_nonsil_like.$x.log  || exit 1

        if [ ! -z `perl -e "print \"true\" if ($sound_like + $speech_like < $nonsil_like)"` ]; then
          goto_phase3=true
        fi
      fi
    fi
  fi

  if $goto_phase3; then
    speech_num_gauss=$speech_num_gauss_init_phase3
    sil_num_gauss=$sil_num_gauss_init_phase3

    $cmd $phase3_dir/log/$utt_id.compute_silence_likes.bootstrap.log \
      gmm-global-get-frame-likes $init_silence_model "$feats" \
      ark:$dir/$utt_id.silence_log_likes.bootstrap.ark || exit 1
    
    $cmd $phase3_dir/log/$utt_id.compute_speech_likes.bootstrap.log \
      gmm-global-get-frame-likes $init_speech_model "$feats" \
      ark:$dir/$utt_id.speech_log_likes.bootstrap.ark || exit 1
  
    cp $tmpdir/$utt_id.vad.bootstrap.ark $phase3_dir/$utt_id.vad.0.ark 

    x=0
    goto_phase3=false

    while [ $x -lt $num_iters_phase3 ]; do
      num_frames_silence=$[num_frames_init_silence + sil_num_gauss * frames_per_gaussian ] 

      if [ $x -lt 3 ]; then
        $cmd $phase3_dir/log/$utt_id.select_top.second.$[x+1].log \
          segmentation-copy --filter-label=0 \
          --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
          ark:$phase3_dir/$utt_id.vad.$x.ark ark:- \| \
          segmentation-select-top --num-bins=$num_bins \
          --merge-dst-label=0 \
          --num-top-frames=-1 --num-bottom-frames=$num_frames_silence \
          --top-select-label=-1 --bottom-select-label=0 --reject-label=1000 \
          --remove-rejected-frames=true \
          --window-size=$window_size --min-window-remainder=$[window_size/2] \
          ark:- ark:$dir/$utt_id.log_energies.ark \
          ark:$phase3_dir/$utt_id.vad.second.$[x+1].ark || exit 1

      else
        $cmd $phase3_dir/log/$utt_id.select_top.$[x+1].log \
          segmentation-copy --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
          --filter-label=0 ark:$phase3_dir/$utt_id.vad.$x.ark \
          ark:$phase3_dir/$utt_id.vad.second.$[x+1].ark || exit 1
      fi

      if [ $x -eq 0 ]; then
        {
          cat $dir/trans.mdl;
          echo "<DIMENSION> $feat_dim <NUMPDFS> 2";
          select-feats-from-segmentation --select-label=0 "$feats" \
            ark:$phase3_dir/$utt_id.vad.second.$[x+1].ark ark:- | \
            gmm-global-init-from-feats --binary=false \
            --num-iters=$[sil_num_gauss+1] --num-gauss-init=1 --num-gauss=$sil_num_gauss \
            ark:- - || exit 1
          gmm-global-copy --binary=false $init_speech_model - || exit 1
        } 2> $phase3_dir/log/$utt_id.init_gmm.log | \
          gmm-copy - $phase3_dir/$utt_id.$[x+1].mdl 2>> $phase3_dir/log/$utt_id.init_gmm.log || exit 1
      else
        $cmd $phase3_dir/log/$utt_id.gmm_update.$[x+1].log \
          gmm-update-segmentation --pdfs=0 \
          --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\" |" \
          $phase3_dir/$utt_id.$x.mdl "$feats" \
          ark:$phase3_dir/$utt_id.vad.second.$[x+1].ark \
          $phase3_dir/$utt_id.$[x+1].mdl || exit 1
      fi
    
      $cmd $phase3_dir/log/$utt_id.get_seg.$[x+1].log \
        gmm-decode-simple --allow-partial=$allow_partial \
        --word-symbol-table=$dir/graph_2class/words.txt \
        $phase3_dir/$utt_id.$[x+1].mdl $dir/graph_2class/HCLG.fst \
        "$feats" ark:/dev/null ark:- \| \
        ali-to-pdf $phase3_dir/$utt_id.$[x+1].mdl ark:- ark:- \| \
        segmentation-init-from-ali ark:- \
        ark:$phase3_dir/$utt_id.vad.$[x+1].ark || exit 1

      if [ $sil_num_gauss -lt $sil_max_gauss ]; then
        sil_num_gauss=$[sil_num_gauss + sil_gauss_incr]
      fi

      x=$[x+1]
    done    ## Done training Silence and Speech GMMs

    $cmd $phase3_dir/log/$utt_id.init_speech.log \
      segmentation-copy --filter-rspecifier=ark:$tmpdir/$utt_id.vad.bootstrap.ark \
      --filter-label=1 ark:$phase3_dir/$utt_id.vad.$x.ark ark:- \| \
      select-feats-from-segmentation --select-label=1 "$feats" \
        ark:- ark:- \| \
        gmm-global-init-from-feats \
        --num-iters=$[speech_num_gauss+1] --num-gauss-init=1 --num-gauss=$speech_num_gauss \
        ark:- $phase3_dir/$utt_id.speech.$x.mdl

    $cmd $phase3_dir/log/$utt_id.init_gmm.log \
      gmm-init-pdf-from-global $phase3_dir/$utt_id.$x.mdl 1 \
      $phase3_dir/$utt_id.speech.$x.mdl $phase3_dir/$utt_id.$[x+1].mdl || exit 1

    x=$[x+1]

    while [ $x -lt $[num_iters_phase4 + num_iters_phase3+1] ]; do
      if [ $sil_num_gauss -lt $sil_max_gauss_phase4 ]; then
        sil_num_gauss=$[sil_num_gauss + sil_gauss_incr_phase4]
      fi

      if [ $speech_num_gauss -lt $speech_max_gauss_phase4 ]; then
        speech_num_gauss=$[speech_num_gauss + speech_gauss_incr_phase4]
      fi

      $cmd $phase3_dir/log/$utt_id.get_seg.$x.log \
        gmm-decode-simple --allow-partial=$allow_partial \
        --word-symbol-table=$dir/graph_2class/words.txt \
        $phase3_dir/$utt_id.$x.mdl $dir/graph_2class/HCLG.fst \
        "$feats" ark:/dev/null ark:- \| \
        ali-to-pdf $phase3_dir/$utt_id.$x.mdl ark:- ark:- \| \
        segmentation-init-from-ali ark:- \
        ark:$phase3_dir/$utt_id.vad.$x.ark || exit 1
      
      $cmd $phase3_dir/log/$utt_id.gmm_update.$[x+1].log \
        gmm-update-segmentation \
        --mix-up-rxfilename="echo -e \"0 $sil_num_gauss\n1 $speech_num_gauss\" |" \
        $phase3_dir/$utt_id.$x.mdl "$feats" \
        ark:$phase3_dir/$utt_id.vad.$x.ark \
        $phase3_dir/$utt_id.$[x+1].mdl || exit 1
      
      x=$[x+1]
    done  ## Done training all 3 GMMs

    cp $phase3_dir/$utt_id.$x.mdl $dir/$utt_id.final.mdl
    rm -f $dir/$utt_id.graph_final
    ln -s graph_2class_test_${speech_to_sil_ratio}x $dir/$utt_id.graph_final
  fi

  $cmd $dir/log/$utt_id.get_seg.final.log \
    gmm-decode-simple --allow-partial=$allow_partial \
    --word-symbol-table=$dir/$utt_id.graph_final/words.txt \
    $dir/$utt_id.final.mdl $dir/$utt_id.graph_final/HCLG.fst \
    "$feats" ark:/dev/null ark:- \| \
    ali-to-pdf $dir/$utt_id.final.mdl ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    ark:$dir/$utt_id.vad.final.ark || exit 1

done < $data/feats.scp

