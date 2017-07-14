#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# Features configs (Must match the features used to train the models
# $sat_model_dir and $model_dir)

lang=data/lang   # Must match the one used to train the models
lang_test=data/lang  # Lang directory for decoding.

data_dir=data/train 
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
sat_model_dir=exp/tri5_cleaned
# Model direcotry used to decode the whole-recording version of the $data_dir to
# get target labels for training SAD. This should typically be a 
# speaker-independent system like LDA+MLLT system.
model_dir=exp/tri4
graph_dir=    # If not provided, a new one will be created using $lang_test

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

nstage=-10
train_stage=-10

affix=_1a
stage=-1
nj=80

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. ./lang.conf || exit 1;

. path.sh
. cmd.sh 

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
mkdir -p $dir

# See $lang/phones.txt and decide which should be garbage
garbage_phones="<oov> <vns>"
for p in $garbage_phones; do 
  for affix in "" "_B" "_E" "_I" "_S"; do
    echo "$p$affix"
  done
done > $dir/garbage_phones.txt

silence_phones="<sss> SIL"
for p in $silence_phones; do 
  for affix in "" "_B" "_E" "_I" "_S"; do
    echo "$p$affix"
  done
done > $dir/silence_phones.txt

if ! cat $dir/garbage_phones.txt $dir/silence_phones.txt | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $dir/{silence,garbage}_phones.txt"
  exit 1
fi

# Create new data directory inside the segmentation directory
data_id=$(basename $data_dir)

whole_data_dir=${data_dir}_whole
whole_data_id=${data_id}_whole

if [ $stage -le 0 ]; then
  rm -r $dir/$data_id || true
  mkdir -p $dir/$data_id

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

  utils/fix_data_dir.sh $dir/$data_id || exit 1

  utils/data/convert_data_dir_to_whole.sh ${data_dir} ${whole_data_dir}

  utils/validate_data_dir.sh --no-feats $dir/$data_id || exit 1
  utils/validate_data_dir.sh --no-text --no-feats $whole_data_dir || exit 1
fi 

data_dir=$dir/${data_id}

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ $stage -le 1 ]; then
  if $use_pitch; then
    steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $nj --write-utt2num-frames true \
      ${whole_data_dir} || exit 1
  else
    steps/make_plp.sh --cmd "$train_cmd" --nj $nj --write-utt2num-frames true \
      ${whole_data_dir} || exit 1
  fi
fi

###############################################################################
# Get feats for the manual segments
###############################################################################
if [ $stage -le 2 ]; then
  utils/data/subsegment_data_dir.sh $whole_data_dir ${data_dir}/segments ${data_dir}/tmp
  cp $data_dir/tmp/feats.scp $data_dir
  awk '{print $1" "$2}' $data_dir/segments > $data_dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt

  steps/compute_cmvn_stats.sh ${data_dir}
  utils/fix_data_dir.sh $data_dir

  # Copy the CMVN stats to the whole directory
  cp $data_dir/cmvn.scp $whole_data_dir   
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

sat_model_id=`basename $sat_model_dir`
###############################################################################
# Get frame-level targets from lattices for nnet training
# Targets are matrices of 3 columns -- silence, speech and garbage
# The target values are obtained by summing up posterior probabilites of 
# arcs from lattice-arc-post over silence, speech and garbage phones.
###############################################################################
if [ $stage -le 6 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones $dir/silence_phones.txt \
    --garbage-phones $dir/garbage_phones.txt \
    --max-phone-duration 0.5 \
    $data_dir $lang $sup_lats_dir \
    $dir/${sat_model_id}_${data_id}_sup_targets
fi

if [ $stage -le 7 ]; then
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones $dir/silence_phones.txt \
    --garbage-phones $dir/garbage_phones.txt \
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
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj 40 \
    $data_dir $whole_data_dir \
    $dir/${sat_model_id}_${data_id}_sup_targets \
    $dir/${sat_model_id}_${whole_data_id}_sup_targets
  
  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj 40 3 \
    $whole_data_dir \
    $dir/${sat_model_id}_${whole_data_id}_sup_targets \
    $dir/${sat_model_id}_${whole_data_id}_sup_targets_sub3
fi

###############################################################################
# Convert the targets from decoding to whole recording. 
###############################################################################
if [ $stage -le 9 ]; then
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj 40 \
    $dir/${uniform_seg_data_id} $whole_data_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets

  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj 40 3 \
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
    --nj 40 --frame-subsampling-factor 3 \
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
  steps/segmentation/merge_targets_dirs.sh --cmd "$train_cmd" --nj 40 \
    --weights $merge_weights --remove-mismatch-frames true \
    $whole_data_dir \
    $dir/${sat_model_id}_${whole_data_id}_sup_targets_sub3 \
    $dir/${model_id}_${whole_data_id}_targets_sub3 \
    $dir/out_of_seg_${whole_data_id}_default_targets_sub3 \
    $dir/${whole_data_id}_combined_targets_sub3
fi

if [ $stage -le 12 ]; then
  utils/copy_data_dir.sh ${whole_data_dir} ${whole_data_dir}_hires_bp
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf --nj 40 \
    ${whole_data_dir}_hires_bp
  steps/compute_cmvn_stats.sh ${whole_data_dir}_hires_bp
fi

if [ $stage -le 13 ]; then
  # Train a TDNN-LSTM network for SAD
  local/segmentation/tuning/train_lstm_asr_sad_1a.sh \
    --stage $nstage --train-stage $train_stage \
    --targets-dir $dir/${whole_data_id}_combined_targets_sub3 \
    --data-dir ${whole_data_dir}_hires_bp
fi

if [ $stage -le 14 ]; then
  # The options to this script must match the options used in the 
  # nnet training script. 
  # e.g. extra-left-context is 70, because the model is an LSTM trained with a 
  # chunk-left-context of 60. 
  # Note: frames-per-chunk is 150 even though the model was trained with 
  # chunk-width of 20. This is just for speed.
  # See the script for details of the options.
  steps/segmentation/detect_speech_activity.sh \
    --extra-left-context 70 --extra-right-context 0 --frames-per-chunk 150 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --nj 32 --acwt 0.3 \
    data/dev10h.pem \
    exp/segmentation_1a/tdnn_lstm_asr_sad_1a \
    mfcc_hires_bp \
    exp/segmentation_1a/tdnn_lstm_asr_sad_1a/{,dev10h}
fi
