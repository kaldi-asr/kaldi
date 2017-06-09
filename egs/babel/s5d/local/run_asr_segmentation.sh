#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# Features configs
feat_type=plp
feat_config=conf/plp.conf
add_pitch=true
pitch_config=conf/pitch.conf

# Uniform segmentation options
max_segment_duration=10
overlap_duration=2.5
max_remaining_duration=5

# Decoding options
lang=data/lang
lang_test=data/lang

data_dir=data/train
sat_model_dir=/export/b17/jtrmal/babel/104-pashto-flp80-p-ext/exp/tri5_cleaned
model_dir=/export/b17/jtrmal/babel/104-pashto-flp80-p-ext/exp/tri4
graph_dir=

affix=_1a
stage=-1
nj=80

. path.sh
. cmd.sh 

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
mkdir -p $dir

cat <<EOF > $dir/garbage_words.txt
<hes>
<noise>
<v-noise>
EOF

cat <<EOF > $dir/silence_words.txt
<silence>
EOF

function make_mfcc {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$train_cmd
  local pitch_config=$pitch_config

  while [ $# -gt 0 ]; do 
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    elif [ $1 == "--mfcc-config" ]; then
      mfcc_config=$2
      shift; shift;
    elif [ $1 == "--add-pitch" ]; then
      add_pitch=$2
      shift; shift;
    elif [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    elif [ $1 == "--pitch-config" ]; then
      pitch_config=$2
      shift; shift;
    else
      break
    fi
  done

  if [ $# -ne 3 ]; then
    echo "Usage: make_mfcc <data-dir> <temp-dir> <feat-dir>"
    exit 1
  fi

  if $add_pitch; then
    steps/make_mfcc_pitch.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --mfcc-config $mfcc_config --pitch-config $pitch_config $* || exit 1
  else
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --mfcc-config $mfcc_config $* || exit 1
  fi

}

function make_plp {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$train_cmd
  local pitch_config=$pitch_config
  
  while [ $# -gt 0 ]; do 
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    elif [ $1 == "--plp-config" ]; then
      plp_config=$2
      shift; shift;
    elif [ $1 == "--add-pitch" ]; then
      add_pitch=$2
      shift; shift;
    elif [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    elif [ $1 == "--pitch-config" ]; then
      pitch_config=$2
      shift; shift;
    else
      break
    fi
  done

  if [ $# -gt 3 ]; then
    echo "Usage: make_plp <data-dir> <temp-dir> <feat-dir>"
    exit 1
  fi
  
  if $add_pitch; then
    steps/make_plp_pitch.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --plp-config $plp_config --pitch-config $pitch_config $* || exit 1
  else
    steps/make_plp.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --plp-config $plp_config $1 $* || exit 1
  fi
}

# Create new data directory inside the segmentation directory
data_id=$(basename $data_dir)
data_dir=$dir/${data_id}

whole_data_dir=${data_dir}_whole
whole_data_id=${data_id}_whole

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh $data_dir $dir/${data_id}
  utils/data/convert_data_dir_to_whole.sh ${data_dir} ${whole_data_dir}
fi 

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ $stage -le 1 ]; then
  if [ $feat_type == "mfcc" ]; then
    make_mfcc --cmd "$train_cmd --max-jobs-run 40" --nj $nj \
      --mfcc-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${whole_data_dir} || exit 1
  elif [ $feat_type == "plp" ]; then
    make_plp --cmd "$train_cmd --max-jobs-run 40" --nj $nj \
      --plp-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${whole_data_dir} || exit 1
  else
    echo "$0: Unknown feat-type $feat_type. Must be mfcc or plp."
    exit 1
  fi
  steps/compute_cmvn_stats.sh \
    ${whole_data_dir} || exit 1
      
  utils/fix_data_dir.sh ${whole_data_dir}
fi

###############################################################################
# Get feats for the manual segments
###############################################################################
if [ $stage -le 2 ]; then
  utils/data/subsegment_data_dir.sh $whole_data_dir ${data_dir}/segments ${data_dir}/tmp
  cp $data_dir/tmp/feats.scp $data_dir
  steps/compute_cmvn_stats.sh ${data_dir}
  utils/fix_data_dir.sh $data_dir
fi

###############################################################################
# Obtain supervision-constrained lattices
###############################################################################
sup_lats_dir=$dir/`basename ${sat_model_dir}`_sup_lats_${data_id}
if [ $stage -le 2 ]; then
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir} ${lang} ${sat_model_dir} $sup_lats_dir || exit 1
fi

###############################################################################
# Uniformly segment whole data directory for decoding
###############################################################################
uniform_seg_data_dir=$dir/${whole_data_id}_uniformseg_${max_segment_duration}sec
uniform_seg_data_id=`basename $uniform_seg_data_dir`

if [ $stage -le 3 ]; then
  utils/data/get_segments_for_data.sh ${whole_data_dir} > \
    ${whole_data_dir}/segments

  mkdir -p $uniform_seg_data_dir

  utils/data/get_uniform_subsegments.py \
    --max-segment-duration $max_segment_duration \
    --overlap-duration $overlap_duration \
    --max-remaining-duration $max_remaining_duration \
    ${whole_data_dir}/segments > $uniform_seg_data_dir/sub_segments

  utils/data/subsegment_data_dir.sh $whole_data_dir \
    $uniform_seg_data_dir/sub_segments $uniform_seg_data_dir
  awk '{print $1" "$1}' $uniform_seg_data_dir/segments > $uniform_seg_data_dir/utt2spk
  cp $uniform_seg_data_dir/utt2spk $uniform_seg_data_dir/spk2utt
  steps/compute_cmvn_stats.sh $uniform_seg_data_dir
fi

model_id=$(basename $model_dir)
###############################################################################
# Create graph dir for decoding
###############################################################################
if [ -z "$graph_dir" ]; then
  graph_dir=$dir/$model_id/graph
  if [ $stage -le 4 ]; then
    if [ ! -f $graph_dir/HCLG.fst ]; then
      cp -rT $lang_test/ $dir/lang_test
      utils/mkgraph.sh $dir/lang_test $model_dir $graph_dir || exit 1
    fi
  fi
fi

###############################################################################
# Decode uniformly segmented data directory
###############################################################################
model_id=$(basename $model_dir)
decode_dir=$dir/${model_id}/decode_${uniform_seg_data_id}
if [ $stage -le 5 ]; then 
  mkdir -p $decode_dir
  
  cp $model_dir/{final.mdl,final.mat,*_opts,tree} $dir/${model_id}
  cp $model_dir/phones.txt $dir/$model_id

  steps/decode.sh --cmd "$decode_cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 \
    --decode-extra-opts "--word-determinize=false" --skip-scoring true \
    $graph_dir $uniform_seg_data_dir $decode_dir
fi

###############################################################################
# Get targets from lattices
###############################################################################
if [ $stage -le 6 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-words $dir/silence_words.txt \
    --garbage-words $dir/garbage_words.txt \
    --max-phone-duration 0.5 \
    $data_dir $lang $sup_lats_dir \
    $dir/${model_id}_${data_id}_sup_targets
fi

if [ $stage -le 7 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-words $dir/silence_words.txt \
    --garbage-words $dir/garbage_words.txt \
    --max-phone-duration 0.5 \
    $uniform_seg_data_dir $lang $decode_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets
fi

###############################################################################
# Convert targets to be w.r.t. whole data directory
###############################################################################
if [ $stage -le 8 ]; then
  echo " [ 1 0 0 ]" > $dir/${model_id}_${whole_data_id}_sup_targets/default_targets.vec
  
  steps/segmentation/convert_targets_dir_to_whole.sh --cmd "$train_cmd" --nj 40 \
    --default-targets $dir/${model_id}_${whole_data_id}_sup_targets/default_targets.vec \
    $data_dir $whole_data_dir \
    $dir/${model_id}_${data_id}_sup_targets \
    $dir/${model_id}_${whole_data_id}_sup_targets
  
  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj 40 3 \
    $whole_data_dir \
    $dir/${model_id}_${whole_data_id}_sup_targets \
    $dir/${model_id}_${whole_data_id}_sup_targets_sub3
fi

###############################################################################
# Resample targets to required output-subsampling rate
###############################################################################
if [ $stage -le 9 ]; then
  steps/segmentation/convert_targets_dir_to_whole.sh --cmd "$train_cmd" --nj 40 \
    $dir/${uniform_seg_data_id} $whole_data_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets

  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj 40 3 \
    $whole_data_dir \
    $dir/${model_id}_${whole_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets_sub3
fi

merge_weights=1.0,0.1
###############################################################################
# Merge targets for the same data from multiple sources (systems)
###############################################################################
if [ $stage -le 10 ]; then
  steps/segmentation/merge_targets_dirs.py --cmd "$train_cmd" --nj 40 \
    --weights $merge_weights --remove-mismatch-frames=true \
    $whole_data_dir \
    $dir/${model_id}_${whole_data_id}_sup_targets_sub3 \
    $dir/${model_id}_${whole_data_id}_targets_sub3 \
    $dir/${model_id}_${whole_data_id}_combined_targets_sub3
fi


mkdir -p data/lang_asr_sad_fs3_simple
cat <<EOF > data/lang_asr_sad_fs3_simple/classes_info.txt
1 0.8 0.99 10 2:0.009 -1:0.001
2 0.2 0.99 10 1:0.009 -1:0.001
EOF

local/segmentation/tuning/train_lstm_asr_sad_1a2.sh 

steps/segmentation/do_asr_sad_data_dir.sh \
  --extra-left-context 70 --extra-right-context 0 --frames-per-chunk 150 \
  --nj 32 --subsampling-factor 1 \
  --transition-scale 1.0 --loopscale 0.3 --acwt 3 \
  --garbage-in-sil-weight 0 \
  --garbage-in-speech-weight 0.0 \
  --sil-in-speech-weight 0 \
  data/dev10h.pem \
  exp/segmentation_1a/tdnn_lstm_asr_sad_1a2 \
  data/lang_asr_sad_fs3_simple/classes_info.txt \
  mfcc_hires_bp \
  exp/segmentation_1a/tdnn_lstm_asr_sad_1a2/{,dev10h}
