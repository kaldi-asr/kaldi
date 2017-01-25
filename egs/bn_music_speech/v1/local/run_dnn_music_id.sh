#! /bin/bash

set -e 
set -o pipefail 
set -u

stage=-1
segmentation_config=conf/segmentation.conf
cmd=run.pl
nj=40

# Viterbi options
min_silence_duration=3   # minimum number of frames for silence
min_speech_duration=3   # minimum number of frames for speech
min_music_duration=3    # minimum number of frames for music
frame_subsampling_factor=1
music_transition_probability=0.1
sil_transition_probability=0.1
speech_transition_probability=0.1
sil_prior=0.3
speech_prior=0.4
music_prior=0.3

# Decoding options
acwt=1
beam=10
max_active=7000

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data> <sad-likes-dir> <music-likes-dir> <segmentation-dir> <music-segmentation-dir> <dir>"
  echo " e.g.: $0 data/bn exp/nnet3_sad_snr/tdnn_b_n4/sad_bn_whole exp/nnet3_sad_snr/tdnn_b_n4/music_bn_whole exp/nnet3_sad_snr/tdnn_b_n4/segmentation_bn_whole exp/nnet3_sad_snr/tdnn_b_n4/segmentation_music_bn_whole exp/dnn_music_id"
  exit 1
fi

data=$1
sad_likes_dir=$2
music_likes_dir=$3
dir=$4

min_silence_duration=`perl -e "print (int($min_silence_duration / $frame_subsampling_factor))"`
min_speech_duration=`perl -e "print (int($min_speech_duration / $frame_subsampling_factor))"`
min_music_duration=`perl -e "print (int($min_music_duration / $frame_subsampling_factor))"`

lang=$dir/lang

if [ $stage -le 1 ]; then
  mkdir -p $lang

  # Create a lang directory with phones.txt and topo with 
  # silence, music and speech phones.
  steps/segmentation/internal/prepare_sad_lang.py \
    --phone-transition-parameters="--phone-list=1 --min-duration=$min_silence_duration --end-transition-probability=$sil_transition_probability" \
    --phone-transition-parameters="--phone-list=2 --min-duration=$min_speech_duration --end-transition-probability=$speech_transition_probability" \
    --phone-transition-parameters="--phone-list=3 --min-duration=$min_music_duration --end-transition-probability=$music_transition_probability" \
    $lang

  cp $lang/phones.txt $lang/words.txt
fi

feat_dim=2    # dummy. We don't need this.
if [ $stage -le 2 ]; then
  $cmd $dir/log/create_transition_model.log gmm-init-mono \
    $lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
fi

# Make unigram G.fst
if [ $stage -le 3 ]; then
  cat > $lang/word2prior <<EOF
1 $sil_prior
2 $speech_prior
3 $music_prior
EOF
  steps/segmentation/internal/make_G_fst.py --word2prior-map $lang/word2prior | \
    fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
    > $lang/G.fst
fi

graph_dir=$dir/graph_test

if [ $stage -le 4 ]; then
  $cmd $dir/log/make_vad_graph.log \
    steps/segmentation/internal/make_sad_graph.sh --iter trans \
    $lang $dir $dir/graph_test || exit 1
fi

if [ $stage -le 5 ]; then
  utils/split_data.sh $data $nj
  sdata=$data/split$nj

  nj_sad=`cat $sad_likes_dir/num_jobs`
  sad_likes=
  for n in `seq $nj_sad`; do
    sad_likes="$sad_likes $sad_likes_dir/log_likes.$n.gz"
  done
  
  nj_music=`cat $music_likes_dir/num_jobs`
  music_likes=
  for n in `seq $nj_music`; do
    music_likes="$music_likes $music_likes_dir/log_likes.$n.gz"
  done

  decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    paste-feats "ark:gunzip -c $sad_likes | extract-feature-segments ark,s,cs:- $sdata/JOB/segments ark:- |" \
    "ark,s,cs:gunzip -c $music_likes | extract-feature-segments ark,s,cs:- $sdata/JOB/segments ark:- | select-feats 1 ark:- ark:- |" \
    ark:- \| decode-faster-mapped  ${decoder_opts[@]} \
    $dir/trans.mdl $graph_dir/HCLG.fst ark:- \
    ark:/dev/null ark:- \| \
    ali-to-phones --per-frame $dir/trans.mdl ark:- \
    "ark:|gzip -c > $dir/ali.JOB.gz"
fi

include_silence=true
if [ $stage -le 6 ]; then
  $cmd JOB=1:$nj $dir/log/get_class_id.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz |" ark:- \| \
    post-to-feats --post-dim=4 ark:- ark:- \| \
    matrix-sum-rows --do-average ark:- ark,t:- \| \
    sid/vector_to_music_labels.pl ${include_silence:+--include-silence-in-music} '>' $dir/ratio.JOB
fi

for n in `seq $nj`; do 
  cat $dir/ratio.$n
done > $dir/ratio

cat $dir/ratio | local/print_scores.py /dev/stdin | compute-eer -
