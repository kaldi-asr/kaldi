#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

stage=-1
train_cmd=run.pl
decode_cmd=run.pl
nj=4
reco_nj=4

lang_test=    # If different from $lang
graph_dir=    # If not provided, a new one will be created using $lang_test

garbage_phones_list=
silence_phones_list=

# Uniform segmentation options for decoding whole recordings. All values are in
# seconds.
max_segment_duration=10
overlap_duration=2.5
max_remaining_duration=5  # If the last remaining piece when splitting uniformly
                          # is smaller than this duration, then the last piece 
                          # is  merged with the previous.

# List of weights on labels obtained from alignment, 
# labels obtained from decoding and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

[ -f . path.sh ] && . ./path.sh 

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 6 ]; then
  cat <<EOF
  This script prepares targets for training neural network for 
  speech activity detction. The targets are obtained from a combination
  of supervision-constrained lattices and lattices obtained by decoding. 

  Usage: $0 <lang> <data> <whole-recording-data> <ali-model-dir> <model-dir> <dir>
   e.g.: $0 data/lang data/train data/train_whole exp/tri5 exp/tri4 exp/segmentation_1a
  
  Note: Both <data> and <whole-recording-data> must have the recording-id
  as speaker, and must contain feats.scp.
EOF
  exit 1
fi

lang=$1   # Must match the one used to train the models
data_dir=$2
whole_data_dir=$3
ali_model_dir=$4  # Model directory used to align the $data_dir to get target 
                  # labels for training SAD. This should typically be a
                  # speaker-adapted system.
model_dir=$5      # Model direcotry used to decode the whole-recording version
                  # of the $data_dir to get target labels for training SAD. This
                  # should typically be a speaker-independent system like
                  # LDA+MLLT system.
dir=$6

mkdir -p $dir

if [ -z "$lang_test" ]; then
  lang_test=$lang
fi

extra_files=
if [ -z "$graph_dir" ]; then
  extra_files="$extra_files $lang_test/G.fst $lang_test/phones.txt"
else
  extra_files="$extra_files $graph_dir/HCLG.fst $graph_dir/phones.txt"
fi

for f in $data_dir/feats.scp $whole_data_dir/feats.scp $data_dir/segments \
  $lang/phones.txt $garbage_phones_list $silence_phones_list \
  $ali_model_dir/final.mdl $model_dir/final.mdl $extra_files; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

if ! cat "$garbage_phones_list $silence_phones_list" | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $garbage_phones_list $silence_phones_list"
  exit 1
fi

data_id=$(basename $data_dir)
whole_data_id=$(basename $whole_data_dir)

if [ $stage -le 0 ]; then
  rm -r $dir/$data_id || true
  mkdir -p $dir/$data_id

  utils/copy_data_dir.sh $data_dir $dir/${data_id}

  # Copy the data directory, but treat the recording as the speaker. This
  # is required to get matching speaker information in the whole 
  # recording data directory.
  cp $data_dir/wav.scp $dir/${data_id}/ || exit 1
  cp $data_dir/reco2file_and_channel $dir/${data_id}/ || true
  awk '{print $1" "$2"-"$1}' $data_dir/segments > $dir/${data_id}/old2new.uttmap || exit 1
  utils/apply_map.pl -f 1 $dir/${data_id}/old2new.uttmap < $data_dir/segments > \
    $dir/${data_id}/segments || exit 1
  awk '{print $1" "$2}' $dir/${data_id}/segments > $dir/$data_id/utt2spk || exit 1
  utils/utt2spk_to_spk2utt.pl $dir/$data_id/utt2spk > $dir/$data_id/spk2utt || exit 1
  utils/apply_map.pl -f 1 $dir/${data_id}/old2new.uttmap < $data_dir/text > \
    $dir/${data_id}/text || exit 1
  utils/apply_map.pl -f 1 $dir/${data_id}/old2new.uttmap < $data_dir/feats.scp > \
    $dir/${data_id}/feats.scp || exit 1

  utils/fix_data_dir.sh $dir/$data_id || exit 1
  utils/validate_data_dir.sh $dir/$data_id || exit 1

  steps/compute_cmvn_stats.sh $dir/$data_id
fi 
data_dir=$dir/${data_id}

###############################################################################
# Get feats for the manual segments
###############################################################################
if [ $stage -le 2 ]; then
  utils/copy_data_dir.sh $whole_data_dir $dir/$whole_data_id

  utils/fix_data_dir.sh $dir/$whole_data_id

  # Copy the CMVN stats to the whole directory
  cp $data_dir/cmvn.scp $dir/$whole_data_id
fi
whole_data_dir=$dir/$whole_data_id

###############################################################################
# Obtain supervision-constrained lattices
###############################################################################
sup_lats_dir=$dir/`basename ${ali_model_dir}`_sup_lats_${data_id}
if [ $stage -le 2 ]; then
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir} ${lang} ${ali_model_dir} $sup_lats_dir || exit 1
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
  awk '{print $1" "$2}' $uniform_seg_data_dir/segments > \
    $uniform_seg_data_dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $uniform_seg_data_dir/utt2spk > \
    $uniform_seg_data_dir/spk2utt
  cp $whole_data_dir/cmvn.scp $uniform_seg_data_dir/
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

ali_model_id=`basename $ali_model_dir`
###############################################################################
# Get frame-level targets from lattices for nnet training
# Targets are matrices of 3 columns -- silence, speech and garbage
# The target values are obtained by summing up posterior probabilites of 
# arcs from lattice-arc-post over silence, speech and garbage phones.
###############################################################################
if [ $stage -le 6 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones "$silence_phones_list" \
    --garbage-phones "$garbage_phones_list" \
    --max-phone-duration 0.5 \
    $data_dir $lang $sup_lats_dir \
    $dir/${ali_model_id}_${data_id}_sup_targets
fi

if [ $stage -le 7 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones "$silence_phones_list" \
    --garbage-phones "$garbage_phones_list" \
    --max-phone-duration 0.5 \
    $uniform_seg_data_dir $lang $decode_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets
fi

###############################################################################
# Convert targets to be w.r.t. whole data directory and subsample the 
# targets by a factor of 3.
# Since the targets from transcript-constrained lattices have only values 
# for the manual segments, these are converted to whole recording-levels 
# by inserting [ 0 0 0 ] for the out-of-manual segment regions.
###############################################################################
if [ $stage -le 8 ]; then
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj $reco_nj \
    $data_dir $whole_data_dir \
    $dir/${ali_model_id}_${data_id}_sup_targets \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets
  
  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj $reco_nj 3 \
    $whole_data_dir \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets_sub3
fi

###############################################################################
# Convert the targets from decoding to whole recording. 
###############################################################################
if [ $stage -le 9 ]; then
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj $reco_nj \
    $dir/${uniform_seg_data_id} $whole_data_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets

  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj $reco_nj 3 \
    $whole_data_dir \
    $dir/${model_id}_${whole_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets_sub3
fi

###############################################################################
# "default targets" values for the out-of-manual-segment regions.
# We assume in this setup that this is silence i.e. [ 1 0 0 ].
###############################################################################

if [ $stage -le 10 ]; then
  echo " [ 1 0 0 ]" > $dir/default_targets.vec
  steps/segmentation/get_targets_for_out_of_segments.sh --cmd "$train_cmd" \
    --nj $reco_nj --frame-subsampling-factor 3 \
    --default-targets $dir/default_targets.vec \
    $data_dir $whole_data_dir $dir/out_of_seg_${whole_data_id}_default_targets_sub3
fi

###############################################################################
# Merge targets for the same data from multiple sources (systems)
# --weights is used to weight targets from alignment with a higher weight 
# the targets from decoding. 
# If --remove-mismatch-frames is true, then if alignment and decoding 
# disagree (more than 0.5 probability on different classes), then those frames
# are removed by setting targets to [ 0 0 0 ]. 
###############################################################################
if [ $stage -le 11 ]; then
  steps/segmentation/merge_targets_dirs.sh --cmd "$train_cmd" --nj $reco_nj \
    --weights $merge_weights --remove-mismatch-frames true \
    $whole_data_dir \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets_sub3 \
    $dir/${model_id}_${whole_data_id}_targets_sub3 \
    $dir/out_of_seg_${whole_data_id}_default_targets_sub3 \
    $dir/${whole_data_id}_combined_targets_sub3
fi

cp $dir/${whole_data_id}_combined_targets_sub3/targets.scp $dir/

echo "$0: Prepared targets in $dir/targets.scp"
