#!/bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0 

set -o pipefail

. path.sh

cmd=run.pl
nj=4
silence_weight=0
speech_prior=0.2
sil_prior=0.8
method=Viterbi
stage=-1

# Viterbi options
min_silence_duration=30   # minimum number of frames for silence
min_speech_duration=30    # minimum number of frames for speech

# Decoding options
acwt=1
beam=10
max_active=7000

. utils/parse_options.sh

data_dir=data/dev_aspire_whole_seg_v102
file_vad_dir=exp/vad_dev_aspire_v102
dir=exp/nnet2_multicondition/ivector_weights_dev_aspire_whole_seg_v102

if [ $# -ne 3 ]; then
  echo "Usage: $0 <segmented-data-dir> <file-vad-dir> <dir>"
  echo " e.g.: $0 $data_dir $file_vad_dir $dir"
  exit 1
fi

data_dir=$1
file_vad_dir=$2
dir=$3

for f in $file_vad_dir/log_likes.1.scp; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f" 
    exit 1
  fi
done

utils/split_data.sh $data_dir $nj || exit 1

perl -e "\$sum_prior = $speech_prior + $sil_prior; printf ('[ %f %f ]', log($sil_prior)-log(\$sum_prior), log($speech_prior)-log(\$sum_prior));" > $dir/log_priors

case $method in
  "Weighting")
    $cmd JOB=1:$nj $dir/log/extract_weights.JOB.log \
      extract-feature-segments --snip-edges=true \
      "ark:cat $file_vad_dir/log_likes.*.ark |" \
      ark,t:$data_dir/split$nj/JOB/segments ark:- \| \
      matrix-add-offset ark:- $dir/log_priors ark:- \| \
      logprob-to-post ark:- ark:- \| \
      weight-pdf-post $silence_weight 0 ark:- ark:- \| \
      post-to-weights ark:- "ark:|gzip -c > $dir/weights.JOB.gz" || exit 1
    ;;
  "Viterbi")
    # Prepare a lang directory
    if [ $stage -le 1 ]; then
      mkdir -p $dir/local/dict
      mkdir -p $dir/local/lm

      echo "1" > $dir/local/dict/silence_phones.txt
      echo "1" > $dir/local/dict/optional_silence.txt
      echo "2" > $dir/local/dict/nonsilence_phones.txt
      echo -e "1 1\n2 2" > $dir/local/dict/lexicon.txt
      echo -e "1\n2\n1 2" > $dir/local/dict/extra_questions.txt

      mkdir -p $dir/lang
      diarization/prepare_vad_lang.sh --num-sil-states $min_silence_duration \
        --num-nonsil-states $min_speech_duration \
        $dir/local/dict $dir/local/lang $dir/lang || exit 1
    fi

    feat_dim=2    # dummy. We don't need this.
    if [ $stage -le 2 ]; then
      $cmd $dir/log/create_transition_model.log gmm-init-mono \
        $dir/lang/topo $feat_dim - $dir/tree \| \
        copy-transition-model --binary=false - $dir/trans.mdl || exit 1
    fi

    lang=$dir/lang_test_${t}x
    if [ $stage -le 3 ]; then
      cp -r $dir/lang $lang
      perl -e '$sil_prior = shift @ARGV; $speech_prior = shift @ARGV; $s = $sil_prior + $speech_prior; $sil_prior = $sil_prior / $s; $speech_prior = $speech_prior / $s; $s = $sil_prior + $speech_prior; print "0 0 1 1 " . -log($sil_prior/(1.1 * $s)) . "\n0 0 2 2 ". -log($speech_prior/(1.1 * $s)). "\n0 ". -log(0.1 / 1.1)' $sil_prior $speech_prior | \
        fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
        --keep_isymbols=false --keep_osymbols=false \
        > $lang/G.fst || exit 1
    fi

    if [ $stage -le 4 ]; then
      $cmd $dir/log/make_vad_graph.log \
        diarization/make_vad_graph.sh --iter trans \
        $lang $dir $dir/graph_test_${t}x || exit 1
    fi

    file_nj=`cat $file_vad_dir/num_jobs` || exit 1
    
    log_likes=ark:$file_vad_dir/log_likes.JOB.ark

    decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)

    if [ $stage -le 5 ]; then
      $cmd JOB=1:$file_nj $dir/log/decode.JOB.log \
        decode-faster-mapped ${decoder_opts[@]} \
        $dir/trans.mdl \
        $dir/graph_test_${t}x/HCLG.fst $log_likes \
        ark:/dev/null ark:- \| \
        ali-to-pdf $dir/trans.mdl ark:- \
        "ark:|gzip -c > $dir/ali.JOB.gz" || exit 1
    fi

    if [ $stage -le 6 ]; then
      $cmd JOB=1:$nj $dir/log/extract_weights.JOB.log \
        extract-int-vector-segments --snip-edges=true \
        "ark:gunzip -c $dir/ali.*.gz |" \
        ark,t:$data_dir/split$nj/JOB/segments ark:- \| \
        ali-to-post ark:- ark:- \| \
        weight-pdf-post $silence_weight 0 ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c > $dir/weights.JOB.gz" || exit 1
    fi

    ;;
  *)
    echo "$0: Unknown method $method for weights extraction"
    exit 1
    ;;
esac

for n in `seq $nj`; do cat $dir/weights.$n.gz; done > $dir/weights.gz


