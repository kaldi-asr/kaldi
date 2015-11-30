#! /bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -u
. path.sh

cmd=run.pl
method=Smoothing
stage=-10

# General segmentation options
max_intersegment_length=50  # Merge nearby speech segments if the silence
                            # between them is less than this many frames.
max_relabel_length=10  # maximum duration of speech that will be removed as part
                       # of smoothing process. This is only if there are no other
                       # speech segments nearby.
pad_length=50         # Pad speech segments by this many frames on either side
max_segment_length=1000   # Segments that are longer than this are split into
                          # overlapping frames.
overlap_length=100        # Overlapping frames when segments are split.
                          # See the above option.

# Viterbi options
min_silence_duration=30   # minimum number of frames for silence
min_speech_duration=30    # minimum number of frames for speech
speech_to_sil_ratio=1     # the prior on speech vs silence

# Decoding options
acwt=1
beam=10
max_active=7000

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data-dir> <vad-dir> <segmentation-dir> <segmented-data-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire exp/segmentation_dev_aspire data/dev_aspire_seg"
  exit 1
fi

data_dir=$1
vad_dir=$2
dir=$3
segmented_data_dir=$4

nj=`cat $vad_dir/num_jobs` || exit 1

mkdir -p $dir

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh $data_dir $segmented_data_dir || exit 1
  rm -f $segmented_data_dir/{cmvn.scp,feats.scp,text,segments,utt2spk,spk2utt}
fi

decoder_opts=(--allow-partial=true)
case $method in
  "Smoothing")
    if [ $stage -le 1 ]; then
      cat <<EOF > $dir/prob_to_ali.awk
#!/bin/awk -f
{
  printf \$1;
  for (i=3; i < NF; i++) {
    if (\$i > 0.5)
      printf " 1";
    else
      printf " 0";
    }
    print "";
}
EOF

      $cmd JOB=1:$nj $dir/log/convert_speech_prob_to_segments.JOB.log \
        copy-vector scp:$vad_dir/speech_prob.JOB.scp ark,t:- \| \
        awk -f $dir/prob_to_ali.awk \| \
        segmentation-init-from-ali ark,t:- ark:- \| \
        segmentation-post-process --remove-labels=0 ark:- ark:- \| \
        segmentation-post-process --merge-adjacent-segments=true \
        --max-intersegment-length=$max_intersegment_length ark:- ark:- \| \
        segmentation-post-process --max-relabel-length=$max_relabel_length --relabel-short-segments-class=1 ark:- ark:- \| \
        segmentation-post-process --widen-label=1 --widen-length=$pad_length ark:- ark:- \| \
        segmentation-post-process --merge-adjacent-segments=true \
        --max-intersegment-length=$max_intersegment_length ark:- ark:- \| \
        segmentation-post-process \
        --max-segment-length=$max_segment_length --overlap-length=$overlap_length ark:- ark:- \| \
        segmentation-to-segments ark:- \
        ark,t:$dir/utt2spk.JOB \
        ark,t:$dir/segments.JOB || exit 1
    fi
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

    t=$speech_to_sil_ratio
    lang=$dir/lang_test_${t}x
    if [ $stage -le 3 ]; then
      cp -r $dir/lang $lang
      perl -e '$t = shift @ARGV; print "0 0 1 1 " . -log(1/($t+2)) . "\n0 0 2 2 ". -log($t/($t+2)). "\n0 ". -log(1/($t+2))' $t | \
        fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
        --keep_isymbols=false --keep_osymbols=false \
        > $lang/G.fst || exit 1

    fi

    if [ $stage -le 4 ]; then
      $cmd $dir/log/make_vad_graph.log \
        diarization/make_vad_graph.sh --iter trans \
        $lang $dir $dir/graph_test_${t}x || exit 1
    fi

    log_likes=ark:$vad_dir/log_likes.JOB.ark

    decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)

    if [ $stage -le 5 ]; then
      $cmd JOB=1:$nj $dir/log/decode.JOB.log \
        decode-faster-mapped ${decoder_opts[@]} \
        $dir/trans.mdl \
        $dir/graph_test_${t}x/HCLG.fst $log_likes \
        ark:/dev/null ark:- \| \
        ali-to-phones --per-frame=true $dir/trans.mdl ark:- \
        "ark:|gzip -c > $dir/ali.JOB.gz" || exit 1
    fi

    if [ $stage -le 6 ]; then
      $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
        segmentation-init-from-ali "ark:gunzip -c $dir/ali.JOB.gz |" ark:- \| \
        segmentation-post-process --remove-labels=1 ark:- ark:- \| \
        segmentation-post-process --merge-labels=2 --merge-dst-label=1 --widen-label=1 --widen-length=$pad_length ark:- ark:- \| \
        segmentation-post-process --merge-adjacent-segments=true --max-intersegment-length=$max_intersegment_length ark:- ark:- \| \
        segmentation-post-process --max-segment-length=$max_segment_length --overlap-length=$overlap_length ark:- ark:- \| \
        segmentation-to-segments ark:- \
        ark,t:$dir/utt2spk.JOB \
        ark,t:$dir/segments.JOB || exit 1
    fi
    ;;
  *)
    echo "$0: Unknown method $method specified for segmentation"
    exit 1
esac

for n in `seq $nj`; do
  cat $dir/utt2spk.$n
done > $segmented_data_dir/utt2spk

for n in `seq $nj`; do
  cat $dir/segments.$n
done > $segmented_data_dir/segments

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

utils/utt2spk_to_spk2utt.pl $segmented_data_dir/utt2spk > $segmented_data_dir/spk2utt || exit 1
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi
