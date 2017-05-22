#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

feat_affix=bp_vh
affix=
reco_nj=32

stage=-1

# SAD network config
iter=final
extra_left_context=100            # Set to some large value
extra_right_context=20


# Configs
frame_subsampling_factor=1

min_silence_duration=3   # minimum number of frames for silence
min_speech_duration=3   # minimum number of frames for speech
min_music_duration=3    # minimum number of frames for music
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

mfcc_config=conf/mfcc_hires_bp.conf

echo $* 

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-dir> <sad-nnet-dir> <dir>"
  echo " e.g.: $0 data/bn exp/nnet3_sad_snr/tdnn_j_n4 exp/dnn_music_id"
  exit 1
fi

# Set to true if the test data has > 8kHz sampling frequency.
do_downsampling=true

data_dir=$1
sad_nnet_dir=$2
dir=$3

data_id=`basename $data_dir`

export PATH="$KALDI_ROOT/tools/sph2pipe_v2.5/:$PATH"
[ ! -z `which sph2pipe` ]

for f in $sad_nnet_dir/$iter.raw $sad_nnet_dir/post_output-speech.vec $sad_nnet_dir/post_output-music.vec; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find $f. See the local/segmentation/run_train_sad.sh"
    exit 1
  fi
done

mkdir -p $dir

new_data_dir=$dir/${data_id}
if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir ${new_data_dir}_whole
  
  freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
  sox=`which sox`

  cat $data_dir/wav.scp | python -c "import sys
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[-1] == '|':
    out_line = line.strip() + ' $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'
  else:
    out_line = 'cat {0} {1} | $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'.format(splits[0], ' '.join(splits[1:]))
  print (out_line)" > ${new_data_dir}_whole/wav.scp

  utils/copy_data_dir.sh ${new_data_dir}_whole ${new_data_dir}_whole_bp_hires
fi

test_data_dir=${new_data_dir}_whole_bp_hires

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
    ${new_data_dir}_whole_bp_hires exp/make_hires/${data_id}_whole_bp mfcc_hires
  steps/compute_cmvn_stats.sh ${new_data_dir}_whole_bp_hires exp/make_hires/${data_id}_whole_bp mfcc_hires
fi

if [ $stage -le 2 ]; then
  output_name=output-speech
  post_vec=$sad_nnet_dir/post_${output_name}.vec
  steps/nnet3/compute_output.sh --nj $reco_nj --cmd "$train_cmd" \
    --post-vec "$post_vec" \
    --iter $iter \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk 150 \
    --output-name $output_name \
    --frame-subsampling-factor $frame_subsampling_factor \
    --get-raw-nnet-from-am false ${test_data_dir} $sad_nnet_dir $dir/sad_${data_id}_whole_bp
fi

if [ $stage -le 3 ]; then
  output_name=output-music
  post_vec=$sad_nnet_dir/post_${output_name}.vec
  steps/nnet3/compute_output.sh --nj $reco_nj --cmd "$train_cmd" \
    --post-vec "$post_vec" \
    --iter $iter \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk 150 \
    --output-name $output_name \
    --frame-subsampling-factor $frame_subsampling_factor \
    --get-raw-nnet-from-am false ${test_data_dir} $sad_nnet_dir $dir/music_${data_id}_whole_bp
fi

if [ $stage -le 4 ]; then
  $train_cmd JOB=1:$reco_nj $dir/get_average_likes.JOB.log \
    paste-feats \
    "ark:gunzip -c $dir/sad_${data_id}_whole_bp/log_likes.JOB.gz | extract-feature-segments ark:- 'utils/filter_scp.pl -f 2 ${test_data_dir}/split$reco_nj/JOB/utt2spk $data_dir/segments |' ark:- |" \
    "ark:gunzip -c $dir/music_${data_id}_whole_bp/log_likes.JOB.gz | select-feats 1 ark:- ark:- | extract-feature-segments ark:- 'utils/filter_scp.pl -f 2 ${test_data_dir}/split$reco_nj/JOB/utt2spk $data_dir/segments |' ark:- |" \
    ark:- \| \
    matrix-sum-rows --do-average ark:- ark,t:$dir/average_likes.JOB.ark
    
  for n in `seq $reco_nj`; do
    cat $dir/average_likes.$n.ark
  done | awk '{print $1" "( exp($3) + exp($5) + 0.01) / (exp($4) + 0.01)}' | \
    local/print_scores.py /dev/stdin | compute-eer -
fi

lang=$dir/lang

if [ $stage -le 5 ]; then
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
if [ $stage -le 6 ]; then
  $train_cmd $dir/log/create_transition_model.log gmm-init-mono \
    $lang/topo $feat_dim - $dir/tree \| \
    copy-transition-model --binary=false - $dir/trans.mdl || exit 1
fi

# Make unigram G.fst
if [ $stage -le 7 ]; then
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

if [ $stage -le 8 ]; then
  $train_cmd $dir/log/make_vad_graph.log \
    steps/segmentation/internal/make_sad_graph.sh --iter trans \
    $lang $dir $dir/graph_test || exit 1
fi

seg_dir=$dir/segmentation_${data_id}_whole_bp
mkdir -p $seg_dir

if [ $stage -le 9 ]; then
  decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)
  $train_cmd JOB=1:$reco_nj $dir/decode.JOB.log \
    paste-feats \
    "ark:gunzip -c $dir/sad_${data_id}_whole_bp/log_likes.JOB.gz | extract-feature-segments ark:- 'utils/filter_scp.pl -f 2 ${test_data_dir}/split$reco_nj/JOB/utt2spk $data_dir/segments |' ark:- |" \
    "ark:gunzip -c $dir/music_${data_id}_whole_bp/log_likes.JOB.gz | select-feats 1 ark:- ark:- | extract-feature-segments ark:- 'utils/filter_scp.pl -f 2 ${test_data_dir}/split$reco_nj/JOB/utt2spk $data_dir/segments |' ark:- |" \
    ark:- \| decode-faster-mapped  ${decoder_opts[@]} \
    $dir/trans.mdl $graph_dir/HCLG.fst ark:- \
    ark:/dev/null ark:- \| \
    ali-to-phones --per-frame $dir/trans.mdl ark:- \
    "ark:|gzip -c > $seg_dir/ali.JOB.gz"
fi

include_silence=true
if [ $stage -le 10 ]; then
  $train_cmd JOB=1:$reco_nj $dir/log/get_class_id.JOB.log \
    ali-to-post "ark:gunzip -c $seg_dir/ali.JOB.gz |" ark:- \| \
    post-to-feats --post-dim=4 ark:- ark:- \| \
    matrix-sum-rows --do-average ark:- ark,t:- \| \
    sid/vector_to_music_labels.pl ${include_silence:+--include-silence-in-music} '>' $dir/ratio.JOB

  for n in `seq $reco_nj`; do 
    cat $dir/ratio.$n
  done > $dir/ratio

  cat $dir/ratio | local/print_scores.py /dev/stdin | compute-eer -
fi

# LOG (compute-eer:main():compute-eer.cc:136) Equal error rate is 0.860585%, at threshold 1.99361
