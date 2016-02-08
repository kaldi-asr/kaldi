#!/bin/bash

set -u
set -e 
set -o pipefail

. path.sh

stage=-2
reco_nj=40
nj=100
cmd=queue.pl
map_noise_to_sil=true
map_unknown_to_speech=true
feat_type=mfcc
add_pitch=false
pitch_config=
phone_map=
feat_config=
config_dir=conf
outside_keep_proportion=1.0
get_whole_recordings_and_weights=true

. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "This script takes a data directory and creates a new data directory "
  echo "and speech activity labels "
  echo "for the purpose of training a Universal Speech Activity Detector."
  echo "Usage: $0 [options] <data-dir> <lang> <ali-dir> <model-dir> <out-data-dir> <dir>"
  echo " e.g.: $0 data/train_100k data/lang exp/tri4a_ali_100k exp/vad_data_prep"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --file-nj <#njobs|4>                             # Split a whole data directory into these many pieces"
  echo "  --nj      <#njobs|4>                             # Split a segmented data directory into these many pieces"
  exit 1
fi

data_dir=$1
lang=$2
ali_dir=$3
model_dir=$4
out_data_dir=$5
dir=$6

if [ $feat_type != "plp" ] && [ $feat_type != "mfcc" ]; then
  echo "$0: --feat-type must be plp or mfcc. Must match the model_dir used."
  exit 1
fi

[ -z "$phone_map" ] && phone_map=$config_dir/phone_map
[ -z "$feat_config" ] && feat_config=$config_dir/$feat_type.conf
[ -z "$pitch_config" ] && pitch_config=$config_dir/pitch.conf

extra_files=

if $add_pitch; then
  extra_files="$extra_files $pitch_config"
fi

for f in $phone_map $feat_config $extra_files; do
  if [ ! -f $f ]; then
    echo "$f could not be found"
    exit 1
  fi
done

mkdir -p $dir

function make_mfcc {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$cmd
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
    steps/make_mfcc_pitch.sh --cmd "$cmd" --nj $nj \
      --mfcc-config $mfcc_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
      --mfcc-config $mfcc_config $1 $2 $3 || exit 1
  fi
}

function make_plp {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$cmd
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

  if [ $# -ne 3 ]; then
    echo "Usage: make_plp <data-dir> <temp-dir> <feat-dir>"
    exit 1
  fi
  
  if $add_pitch; then
    steps/make_plp_pitch.sh --cmd "$cmd" --nj $nj \
      --plp-config $plp_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_plp.sh --cmd "$cmd" --nj $nj \
      --plp-config $plp_config $1 $2 $3 || exit 1
  fi
}

if $map_noise_to_sil || $map_unknown_to_speech; then
  cat $phone_map | \
    awk -v map_noise_to_sil=$map_noise_to_sil -v map_unknown_to_speech=$map_unknown_to_speech \
    '{if ($2 == 2 && map_noise_to_sil == "true") print $1" 0"; 
      else if ($2 == 3 && map_unknown_to_speech) print $1" 1";
      else print $0;}' > \
    $dir/phone_map
  phone_map=$dir/phone_map
fi

data_id=$(basename $data_dir)

utils/split_data.sh --per-reco $data_dir $reco_nj

# Convert alignment for the provided segments into 
# initial speech activity labels
vad_dir=$dir/`basename ${ali_dir}`_vad_${data_id}
if [ $stage -le -1 ]; then
  diarization/convert_ali_to_vad.sh --phone-map $phone_map \
    --cmd "$cmd" \
    $data_dir $lang $ali_dir $vad_dir || exit 1
fi

[ ! -s $vad_dir/vad.scp ] && echo "$0: $vad_dir/vad.scp is empty" && exit 1

# Compute total lengths of each recording 
if [ $stage -le 0 ]; then
  $cmd JOB=1:$reco_nj $dir/log/get_recording_lengths.JOB.log \
    wav-to-duration scp:$data_dir/split$reco_nj/JOB/wav.scp \
    ark,t:- \| awk \'\{print \$1 " " int\(\$2 \* 100\)\}\' '>' $dir/reco_lengths.JOB.ark.txt || exit 1

  for n in `seq $reco_nj`; do 
    cat $dir/reco_lengths.$n.ark.txt
  done | sort -u > $dir/reco_lengths.ark.txt
fi

# Create extended data directory that consists of the provided 
# segments along with the segments outside it.
# This is basically dividing the whole recording into pieces
# consisting of pieces corresponding to the provided segments 
# and outside the provided segments.

# First create the segments outside of the provided segments
extended_data_dir=$dir/${data_id}_extended
if [ $stage -le 1 ]; then
  rm -rf $extended_data_dir
  mkdir -p $extended_data_dir/split$reco_nj
  utils/copy_data_dir.sh $data_dir $extended_data_dir
  for f in cmvn.scp feats.scp text; do
    rm -f $extended_data_dir/$f
  done

  $cmd JOB=1:$reco_nj $dir/log/get_empty_segments.JOB.log \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=1 --ignore-missing=false \
    "ark:segmentation-init-from-lengths --label=0 ark,t:$dir/reco_lengths.JOB.ark.txt ark:- |" \
    "ark:segmentation-init-from-segments $data_dir/split$reco_nj/JOB/segments ark:- |" \
    ark:- \| segmentation-post-process --remove-labels=1 ark:- ark:- \| \
    segmentation-post-process --max-segment-length=1000 --post-process-label=0 \
    ark:- ark:- \| segmentation-to-segments --single-speaker=true --frame-overlap=0 \
    ark:- ark,t:$extended_data_dir/split$reco_nj/utt2spk_empty.JOB \
    ark,t:$extended_data_dir/split$reco_nj/segments_empty.JOB || exit 1
fi

awk '{print $1" "$2"-"$1}' $data_dir/segments > $data_dir/old2new.utt_map

# Combine provided segments with segments outside the provided segments to
# create the extended data directory
if [ $stage -le 2 ] ; then
  for n in `seq $reco_nj`; do
    cat $data_dir/split$reco_nj/$n/segments | \
      utils/apply_map.pl -f 1 $data_dir/old2new.utt_map | \
      cat - $extended_data_dir/split$reco_nj/segments_empty.$n | \
      sort -k1,1 | tee $extended_data_dir/split$reco_nj/segments.$n
  done > $extended_data_dir/segments
  
  awk '{print $1" "$2}' $extended_data_dir/segments > $extended_data_dir/utt2spk
  
  utils/utt2spk_to_spk2utt.pl $extended_data_dir/utt2spk > $extended_data_dir/spk2utt
  utils/fix_data_dir.sh $extended_data_dir
fi

## Create text for the extended data directory
if [ $stage -le 3 ]; then
  mkdir -p $dir/split$reco_nj
  for n in `seq $reco_nj`; do
    cat $extended_data_dir/split$reco_nj/utt2spk_empty.$n | awk '{print $1}' > \
      $extended_data_dir/split$reco_nj/text_empty.$n || exit 1
    cat $data_dir/split$reco_nj/$n/text | \
      utils/apply_map.pl -f 1 $data_dir/old2new.utt_map | \
    cat - $extended_data_dir/split$reco_nj/text_empty.$n | sort -k1,1 || tee $extended_data_dir/split$reco_nj/text.$n
  done > $extended_data_dir/text
  utils/fix_data_dir.sh $extended_data_dir
fi

# Get initial voice activity labels for the outside segments and combine them
# with the voice activity labels for the provided segments.
# The extended voice activity labels are put in $dir/vad/vad.scp
if [ $stage -le 4 ]; then
  mkdir -p $dir/vad

  # We split the initial vad.scp based on recording with the same splits as
  # the other files
  for n in `seq $reco_nj`; do
    utils/filter_scp.pl $data_dir/split$reco_nj/$n/utt2spk $vad_dir/vad.scp | \
      utils/apply_map.pl -f 1 $data_dir/old2new.utt_map > \
      $dir/vad/vad_tmp.$n.scp || exit 1
    [ ! -s $dir/vad/vad_tmp.$n.scp ] && echo "$0: no utterances in $dir/vad/vad_tmp.$n.scp" && exit 1
  done
  
  $cmd JOB=1:$reco_nj $dir/log/get_empty_vad.JOB.log \
    segmentation-init-from-segments --label=0 --per-utt=true \
    $extended_data_dir/split$reco_nj/segments_empty.JOB ark:- \| \
    segmentation-to-ali ark:- ark,scp:$dir/vad/vad_empty.JOB.ark,$dir/vad/vad_empty.JOB.scp

  for n in `seq $reco_nj`; do
    cat $dir/vad/vad_tmp.$n.scp $dir/vad/vad_empty.$n.scp | sort -k 1,1 | tee $dir/vad/vad.$n.scp
  done > $dir/vad/vad.scp

  for n in `seq $reco_nj`; do
    cat $dir/vad/vad_tmp.$n.scp 
  done > $dir/vad/vad_tmp.scp
  
  for n in `seq $reco_nj`; do
    cat $dir/vad/vad_empty.$n.scp 
  done > $dir/vad/vad_empty.scp
fi

# Make features for the extended data directory. 
# At this stage, we can split into larger number of pieces.
if [ $stage -le 6 ]; then
  if [ $feat_type == "mfcc" ]; then
    make_mfcc --cmd "$cmd" --nj $nj \
      --mfcc-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${extended_data_dir} exp/make_mfcc/${data_id}_extended mfcc || exit 1
  elif [ $feat_type == "plp" ]; then
    make_plp --cmd "$cmd" --nj $nj \
      --plp-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${extended_data_dir} exp/make_plp/${data_id}_extended plp || exit 1
  fi
  utils/fix_data_dir.sh $extended_data_dir
  
  # We also create a temporary directory to compute cmvn stats
  # only on the provided segments and copy the stats to the 
  # extended data directory
  temp_data_dir=$dir/${data_id}_temp

  rm -rf $temp_data_dir || true

  awk '{print $2" "$1}' $data_dir/old2new.utt_map > $data_dir/new2old.utt_map
  utils/subset_data_dir.sh --utt-list $data_dir/new2old.utt_map $extended_data_dir $temp_data_dir

  if [ $feat_type == "mfcc" ]; then
    make_mfcc --cmd "$cmd" --nj $nj \
      --mfcc-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${temp_data_dir} exp/make_mfcc/${data_id}_temp mfcc || exit 1
    steps/compute_cmvn_stats.sh \
      ${temp_data_dir} exp/make_mfcc/${data_id}_temp mfcc || exit 1
  elif [ $feat_type == "plp" ]; then
    make_plp --cmd "$cmd" --nj $nj \
      --plp-config $feat_config \
      --add-pitch $add_pitch --pitch-config $pitch_config \
      ${temp_data_dir} exp/make_plp/${data_id}_temp plp || exit 1
    steps/compute_cmvn_stats.sh \
      ${temp_data_dir} exp/make_plp/${data_id}_temp plp || exit 1
  fi
  
  cp ${temp_data_dir}/cmvn.scp $extended_data_dir
  rm -rf $extended_data_dir/split*
fi

# By default, we use word LM. If required, we can think 
# consider phone LM
graph_dir=$model_dir/graph
if [ $stage -le 7 ]; then
  if [ ! -d $graph_dir ]; then
    utils/mkgraph.sh ${lang} $model_dir $graph_dir || exit 1
  fi
fi

# Decode without lattice (get only best path)
if [ $stage -le 8 ]; then
  steps/decode_nolats.sh --cmd "$cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 --write-words false \
    --write-alignments true \
    $graph_dir ${extended_data_dir} \
    ${model_dir}/decode_${data_id}_extended || exit 1
fi

# Get VAD based on the decoded best path
decode_vad_dir=$dir/${model_dir}_decode_vad_${data_id}
if [ $stage -le 9 ]; then
  diarization/convert_ali_to_vad.sh --phone-map $phone_map \
    --cmd "$cmd" --model $model_dir/final.mdl \
    $extended_data_dir $graph_dir \
  $model_dir/decode_${data_id}_extended $decode_vad_dir || exit 1
fi


  for n in `seq $reco_nj`; do
    cat $dir/vad/vad_tmp.$n.scp 
  done > $dir/vad/vad_tmp.scp
  
  for n in `seq $reco_nj`; do
    cat $dir/vad/vad_empty.$n.scp 
  done > $dir/vad/vad_empty.scp
# Intersect the initial VAD with the VAD from the decode
if [ $stage -le 10 ]; then
  vad_scps=()
  mkdir -p $dir/vad/split$nj
  mkdir -p $decode_vad_dir/split$nj
  for n in `seq $nj`; do
    vad_scps+=($dir/vad/split$nj/vad.$n.scp)
  done
  utils/split_scp.pl $dir/vad/vad.scp ${vad_scps[@]}
  
  mkdir -p $dir/intersected_segmentations
  
  # For outside of the provided segments,
  #  * Intersect the initial VAD and the decode VAD and label the mismatch 
  #    regions as class 10, which can be removed later.
  $cmd JOB=1:$nj $dir/log/intersect_empty_segments.JOB.log \
    utils/filter_scp.pl $dir/vad/vad_empty.scp $dir/vad/split$nj/vad.JOB.scp \
    '>' $dir/vad/split$nj/vad_empty.JOB.scp '&&' \
    utils/filter_scp.pl $dir/vad/split$nj/vad_empty.JOB.scp $decode_vad_dir/vad.scp \
    '>' $decode_vad_dir/split$nj/vad_empty.JOB.scp '&&' \
    segmentation-intersect-segments --mismatch-label=10 \
    "ark:segmentation-init-from-ali scp:$dir/vad/split$nj/vad_empty.JOB.scp ark:- |" \
    "ark:segmentation-init-from-ali scp:$decode_vad_dir/split$nj/vad_empty.JOB.scp ark:- |"  \
    ark:- \| segmentation-post-process --remove-labels=10 \
    --merge-adjacent-segments=true --max-intersegment-length=10 ark:- \
    ark,scp:$dir/intersected_segmentations/intersected_segmentations_empty.JOB.ark,$dir/intersected_segmentations/intersected_segmentations_empty.JOB.scp || exit 1

  # For the provided segments,
  #  * For now, just convert the inital VAD into segmentations 
  $cmd JOB=1:$nj $dir/log/intersect_provided_segments.JOB.log \
    utils/filter_scp.pl $dir/vad/vad_tmp.scp $dir/vad/split$nj/vad.JOB.scp \
    '>' $dir/vad/split$nj/vad_tmp.JOB.scp '&&' \
    utils/filter_scp.pl $dir/vad/split$nj/vad_tmp.JOB.scp $decode_vad_dir/vad.scp \
    '>' $decode_vad_dir/split$nj/vad_tmp.JOB.scp '&&' \
    segmentation-intersect-segments --mismatch-label=10 \
    "ark:segmentation-init-from-ali scp:$dir/vad/split$nj/vad_tmp.JOB.scp ark:- |" \
    "ark:segmentation-init-from-ali scp:$decode_vad_dir/split$nj/vad_tmp.JOB.scp ark:- |" \
    ark:- \| segmentation-post-process --remove-labels=10 \
    --merge-adjacent-segments=true --max-intersegment-length=10 ark:- \
    ark,scp:$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.ark,$dir/intersected_segmentations/intersected_segmentations_tmp.JOB.scp || exit 1

  for n in `seq $nj`; do 
    cat $dir/intersected_segmentations/intersected_segmentations_empty.$n.scp 
  done > $dir/intersected_segmentations/intersected_segmentations_empty.scp

  for n in `seq $nj`; do
    cat $dir/intersected_segmentations/intersected_segmentations_tmp.$n.scp 
  done > $dir/intersected_segmentations/intersected_segmentations_tmp.scp
fi

# Optionally select only a small percentage of the outside utterances
# in the final set of utterances. This can be used to balance the amount of 
# speech vs silence. 
empty_copy_cmd="cat $dir/intersected_segmentations/intersected_segmentations_empty.scp"
if [ $outside_keep_proportion != 1.0 ]; then
  nlines=`wc -l $dir/intersected_segmentations/intersected_segmentations_empty.scp` || exit 1
  empty_copy_cmd="utils/subset_scp.pl $nlines $dir/intersected_segmentations/intersected_segmentations_empty.scp"
fi

if [ $stage -le 11 ]; then
  eval $empty_copy_cmd  | \
    cat - $dir/intersected_segmentations/intersected_segmentations_tmp.scp > \
    $dir/intersected_segmentations/final_segmentations_p$outside_keep_proportion.scp || exit 1
fi

if [ $stage -le 12 ]; then
  awk '{print $1" "$2}' $extended_data_dir/segments | \
    utils/utt2spk_to_spk2utt.pl > $extended_data_dir/reco2utt

  mkdir -p $dir/reco_segmentations
  mkdir -p $extended_data_dir/split$reco_nj

  reco2utts=()
  for n in `seq $reco_nj`; do
    reco2utts+=($extended_data_dir/split$reco_nj/reco2utt.$n)
  done
  utils/split_scp.pl $extended_data_dir/reco2utt ${reco2utts[@]}

  $cmd JOB=1:$reco_nj $dir/log/get_reco_segmentation.JOB.log \
    utils/spk2utt_to_utt2spk.pl $extended_data_dir/split$reco_nj/reco2utt.JOB '>' $extended_data_dir/split$reco_nj/utt2reco.JOB '&&' \
    segmentation-combine-segments \
    "scp:utils/filter_scp.pl $extended_data_dir/split$reco_nj/utt2reco.JOB $dir/intersected_segmentations/final_segmentations_p$outside_keep_proportion.scp |" \
    "ark,t:utils/filter_scp.pl $extended_data_dir/split$reco_nj/utt2reco.JOB $extended_data_dir/segments |" \
    ark,t:$extended_data_dir/split$reco_nj/reco2utt.JOB ark:- \| \
    segmentation-post-process --remove-labels=3 --merge-adjacent-segments=true \
    --max-segment-length=1000 ark:- \
    ark:$dir/reco_segmentations/reco_segmentation.JOB.ark || exit 1
  
  mkdir -p $dir/reco_vad

  $cmd JOB=1:$reco_nj $dir/log/get_reco_vad.JOB.log \
    segmentation-to-ali --default-label=4 --lengths="ark,t:cat $dir/reco_lengths.*.ark.txt |" \
    ark:$dir/reco_segmentations/reco_segmentation.JOB.ark \
    ark,scp:$dir/reco_vad/vad.JOB.ark,$dir/reco_vad/vad.JOB.scp || exit 1
fi

if $get_whole_recordings_and_weights; then
  if [ $stage -le 13 ]; then
    rm -rf $dir/final_vad
    mkdir -p $dir/final_vad

    $cmd JOB=1:$reco_nj $dir/log/get_deriv_weights.JOB.log \
      segmentation-post-process --merge-labels=0:1:2:3:4:10 --merge-dst-label=1 \
      ark:$dir/reco_segmentations/reco_segmentation.JOB.ark ark:- \| \
      segmentation-to-ali --default-label=0 --lengths="ark,t:cat $dir/reco_lengths.*.ark.txt |" \
      ark:- ark:- \| ali-to-post ark:- ark:- \| weight-pdf-post 0 0 ark:- ark:- \| \
      post-to-weights ark:- \
      ark,scp:$dir/final_vad/deriv_weights.JOB.ark,$dir/final_vad/deriv_weights.JOB.scp || exit 1
  fi

  rm -rf $out_data_dir
  diarization/convert_data_dir_to_whole.sh $extended_data_dir $out_data_dir
  rm -f $out_data_dir/{feats.scp,cmvn.scp,text}
  
  for n in `seq $reco_nj`; do 
    cat $dir/final_vad/deriv_weights.$n.scp 
  done > $dir/final_vad/deriv_weights.scp
  
  echo "$0: Finished creating corpus for training Universal SAD with deriv weights"
  exit 0
fi

# Split the recording into new segments in the output data directory.
# Create VAD corresponding to the same segments
if [ $stage -le 13 ]; then
  rm -rf $out_data_dir
  utils/copy_data_dir.sh $extended_data_dir $out_data_dir
  rm -f $out_data_dir/{feats.scp,cmvn.scp,text}
  
  mkdir -p $out_data_dir/split$reco_nj

  $cmd JOB=1:$reco_nj $dir/log/split_reco_into_segments.JOB.log \
    segmentation-post-process --merge-labels=0:1:2:3:4:10 --merge-dst-label=1 \
    --max-intersegment-length=10 --max-segment-length=1000 --merge-adjacent-segments \
    ark:$dir/reco_segmentations/reco_segmentation.JOB.ark ark:- \| \
    segmentation-to-segments --single-speaker=true --frame-overlap=0 \
    ark:- ark,t:$out_data_dir/split$reco_nj/utt2spk.JOB \
    ark,t:$out_data_dir/split$reco_nj/segments.JOB || exit 1

  for n in `seq $reco_nj`; do
    cat $out_data_dir/split$reco_nj/segments.$n 
  done > $out_data_dir/segments
  
  for n in `seq $reco_nj`; do
    cat $out_data_dir/split$reco_nj/utt2spk.$n 
  done > $out_data_dir/utt2spk

  utils/utt2spk_to_spk2utt.pl $out_data_dir/utt2spk > $out_data_dir/spk2utt

  mkdir -p $dir/final_vad
  $cmd JOB=1:$reco_nj $dir/log/extract_segment_vad.JOB.log \
    extract-int-vector-segments ark:$dir/reco_vad/vad.JOB.ark \
    ark,t:$out_data_dir/split$reco_nj/segments.JOB \
    ark,scp:$dir/final_vad/vad.JOB.ark,$dir/final_vad/vad.JOB.scp || exit 1

  for n in `seq $reco_nj`; do 
    cat $dir/final_vad/vad.$n.scp
  done > $dir/final_vad/vad.scp
fi

echo "$0: Finished creating corpus for training Universal SAD"
