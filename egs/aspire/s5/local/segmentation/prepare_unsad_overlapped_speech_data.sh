#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

set -e
set -u
set -o pipefail

. path.sh

num_data_reps=5
nj=40
cmd=queue.pl
stage=-1

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <corrupted-data-dir> <orig-corrupted-data-dir> <utt-vad-dir> <temp-dir> <out-labels-dir>"
  echo " e.g.: $0 data/fisher_train_100k_sp_75k_hires_bp data/fisher_train_100k_sp_75k/overlapped_segments_info.txt exp/unsad/make_unsad_fisher_train_100k_sp/tri4_ali_fisher_train_100k_sp_vad_fisher_train_100k_sp exp/unsad/make_overlap_labels/fisher_train_100k_sp_75k overlap_labels"
  exit 1
fi

corrupted_data_dir=$1
orig_corrupted_data_dir=$2
utt_vad_dir=$3
tmpdir=$4
overlap_labels_dir=$5

overlapped_segments_info=$orig_corrupted_data_dir/overlapped_segments_info.txt
corrupted_data_id=`basename $orig_corrupted_data_dir`

for f in $corrupted_data_dir/feats.scp $overlapped_segments_info $utt_vad_dir/sad_seg.scp; do
  [ ! -f $f ] && echo "Could not find file $f" && exit 1
done

overlap_dir=$tmpdir/make_overlap_labels_${corrupted_data_id}
unreliable_dir=$tmpdir/unreliable_${corrupted_data_id}

mkdir -p $unreliable_dir

# make $overlap_labels_dir an absolute pathname.
overlap_labels_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $overlap_labels_dir ${PWD}`

# Combine the VAD from the base recording and the VAD from the overlapping segments
# to create per-frame labels of the number of overlapping speech segments
# Unreliable segments are regions where no VAD labels were available for the
# overlapping segments. These can be later removed by setting deriv weights to 0.

if [ $stage -le 1 ]; then
  for n in `seq $num_data_reps`; do
    cat $utt_vad_dir/sad_seg.scp | \
    awk -v n=$n '{print "ovlp"n"_"$0}'
  done | sort -k1,1 > ${corrupted_data_dir}/sad_seg.scp
  utils/data/get_utt2num_frames.sh $corrupted_data_dir
  utils/split_data.sh ${corrupted_data_dir} $nj

  $cmd JOB=1:$nj $overlap_dir/log/get_overlap_seg.JOB.log \
    segmentation-init-from-additive-signals-info --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    --additive-signals-segmentation-rspecifier=scp:$utt_vad_dir/sad_seg.scp \
    --unreliable-segmentation-wspecifier="ark:| gzip -c > $unreliable_dir/unreliable_seg.JOB.gz" \
    "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${nj}/JOB/utt2spk $corrupted_data_dir/sad_seg.scp |" \
    ark,t:$orig_corrupted_data_dir/overlapped_segments_info.txt ark:- \| \
    segmentation-copy --keep-label=1 ark:- ark:- \| \
    segmentation-get-stats --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    ark:- ark:- ark:/dev/null \| \
    segmentation-init-from-ali ark:- "ark:| gzip -c > $overlap_dir/overlap_seg.JOB.gz"
fi

if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $overlap_dir/log/get_overlapped_speech_labels.JOB.log \
    gunzip -c $overlap_dir/overlap_seg.JOB.gz \| \
    segmentation-post-process --remove-labels=0:1 ark:- ark:- \| \
    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- \
    ark,scp:$overlap_labels_dir/overlapped_speech_labels_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/overlapped_speech_labels_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/overlapped_speech_labels_${corrupted_data_id}.$n.scp
  done > ${corrupted_data_dir}/overlapped_speech_labels.scp
fi

if [ $stage -le 3 ]; then
  # First convert the unreliable segments into a segmentation.
  # Initialize a segmentation from utt2num_frames and set to 0, the regions
  # of unreliable segments. At this stage deriv weights is 1 for all but the
  # unreliable segment regions.
  # Initialize a segmentation from the overlap labels and retain regions where 
  # there is speech from at least one speaker. 
  # Intersect this with the deriv weights segmentation from above. 
  # At this stage deriv weights is 1 for only the regions where there is 
  # at least one speaker and the the overlapping segment is not unreliable. 
  # Convert this to deriv weights.
  $cmd JOB=1:$nj $unreliable_dir/log/get_deriv_weights.JOB.log \
    utils/filter_scp.pl $corrupted_data_dir/split$nj/JOB/utt2spk $corrupted_data_dir/utt2num_frames \| \
    segmentation-init-from-lengths ark,t:- ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=0 --ignore-missing \
    ark:- "ark,s,cs:gunzip -c $unreliable_dir/unreliable_seg.JOB.gz | segmentation-to-segments ark:- - | segmentation-init-from-segments - ark:- |" ark:- \| \
    segmentation-intersect-segments --mismatch-label=0 \
    "ark:gunzip -c $overlap_dir/overlap_seg.JOB.gz | segmentation-post-process --remove-labels=0 --merge-labels=1:2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- |" \
    ark,s,cs:- ark:- \| segmentation-post-process --remove-labels=0 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
    ark,scp:$overlap_labels_dir/deriv_weights_for_overlapped_speech_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/deriv_weights_for_overlapped_speech_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/deriv_weights_for_overlapped_speech_$corrupted_data_id.${n}.scp
  done > $corrupted_data_dir/deriv_weights_for_overlapped_speech.scp
fi

if [ $stage -le 4 ]; then
  # Get only first speaker labels as speech_feat as we are not sure of the energy levels of the other speaker.
  $cmd JOB=1:$nj $overlap_dir/log/get_first_speaker_labels.JOB.log \
    gunzip -c $overlap_dir/overlap_seg.JOB.gz \| \
    segmentation-post-process --remove-labels=0 --merge-labels=1:2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| \
    vector-to-feat ark:- \
    ark,scp:$overlap_labels_dir/speech_feat_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/speech_feat_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/speech_feat_${corrupted_data_id}.$n.scp
  done > ${corrupted_data_dir}/speech_feat.scp
fi

if [ $stage -le 5 ]; then
  $cmd JOB=1:$nj $unreliable_dir/log/get_deriv_weights.JOB.log \
    utils/filter_scp.pl $corrupted_data_dir/split$nj/JOB/utt2spk $corrupted_data_dir/utt2num_frames \| \
    segmentation-init-from-lengths ark,t:- ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=0 --ignore-missing \
    ark:- "ark,s,cs:gunzip -c $unreliable_dir/unreliable_seg.JOB.gz | segmentation-to-segments ark:- - | segmentation-init-from-segments - ark:- |" ark:- \| \
    segmentation-post-process --remove-labels=0 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
    ark,scp:$overlap_labels_dir/deriv_weights_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/deriv_weights_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/deriv_weights_$corrupted_data_id.${n}.scp
  done > $corrupted_data_dir/deriv_weights.scp
fi

exit 0

####exit 1
####
####if [ $stage -le 9 ]; then
####  mkdir -p $overlap_data_dir $unreliable_data_dir
####  cp $orig_corrupted_data_dir/wav.scp $overlap_data_dir
####  cp $orig_corrupted_data_dir/wav.scp $unreliable_data_dir
####
####  # Create segments where there is definitely an overlap.
####  # Assume no more than 10 speakers overlap.
####  $cmd JOB=1:$nj $overlap_dir/log/process_to_segments.JOB.log \
####    segmentation-post-process --remove-labels=0:1 \
####    ark:$overlap_dir/overlap_seg_speed_unperturbed.JOB.ark ark:- \| \
####    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- \| \
####    segmentation-to-segments ark:- ark:$overlap_data_dir/utt2spk.JOB $overlap_data_dir/segments.JOB
####
####  $cmd JOB=1:$nj $overlap_dir/log/get_unreliable_segments.JOB.log \
####    segmentation-to-segments --single-speaker \
####    ark:$unreliable_dir/unreliable_seg_speed_unperturbed.JOB.ark \
####    ark:$unreliable_data_dir/utt2spk.JOB $unreliable_data_dir/segments.JOB
####
####  for n in `seq $nj`; do cat $overlap_data_dir/utt2spk.$n; done > $overlap_data_dir/utt2spk
####  for n in `seq $nj`; do cat $overlap_data_dir/segments.$n; done > $overlap_data_dir/segments
####  for n in `seq $nj`; do cat $unreliable_data_dir/utt2spk.$n; done > $unreliable_data_dir/utt2spk
####  for n in `seq $nj`; do cat $unreliable_data_dir/segments.$n; done > $unreliable_data_dir/segments
####
####  utils/fix_data_dir.sh $overlap_data_dir
####  utils/fix_data_dir.sh $unreliable_data_dir
####
####  if $speed_perturb; then
####    utils/data/perturb_data_dir_speed_3way.sh $overlap_data_dir ${overlap_data_dir}_sp
####    utils/data/perturb_data_dir_speed_3way.sh $unreliable_data_dir ${unreliable_data_dir}_sp
####  fi
####fi
####
####if $speed_perturb; then
####  overlap_data_dir=${overlap_data_dir}_sp
####  unreliable_data_dir=${unreliable_data_dir}_sp
####fi
####
##### make $overlap_labels_dir an absolute pathname.
####overlap_labels_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $overlap_labels_dir ${PWD}`
####
####if [ $stage -le 10 ]; then
####  utils/split_data.sh ${overlap_data_dir} $nj
####
####  $cmd JOB=1:$nj $overlap_dir/log/get_overlap_speech_labels.JOB.log \
####    utils/data/get_reco2utt.sh ${overlap_data_dir}/split${reco_nj}reco/JOB '&&' \
####    segmentation-init-from-segments --shift-to-zero=false \
####    ${overlap_data_dir}/split${reco_nj}reco/JOB/segments ark:- \| \
####    segmentation-combine-segments-to-recordings ark:- ark,t:${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt \
####    ark:- \| \
####    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- \
####    ark,scp:$overlap_labels_dir/overlapped_speech_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/overlapped_speech_${corrupted_data_id}.JOB.scp
####fi
####
####for n in `seq $reco_nj`; do
####  cat $overlap_labels_dir/overlapped_speech_${corrupted_data_id}.$n.scp
####done > ${corrupted_data_dir}/overlapped_speech_labels.scp
####
####if [ $stage -le 11 ]; then
####  utils/data/get_reco2utt.sh ${unreliable_data_dir}
####
####  # First convert the unreliable segments into a recording-level segmentation.
####  # Initialize a segmentation from utt2num_frames and set to 0, the regions
####  # of unreliable segments. At this stage deriv weights is 1 for all but the
####  # unreliable segment regions.
####  # Initialize a segmentation from the VAD labels and retain only the speech segments.
####  # Intersect this with the deriv weights segmentation from above. At this stage
####  # deriv weights is 1 for only the regions where base VAD label is 1 and
####  # the overlapping segment is not unreliable. Convert this to deriv weights.
####  $cmd JOB=1:$reco_nj $unreliable_dir/log/get_deriv_weights.JOB.log\
####    segmentation-init-from-segments --shift-to-zero=false \
####    "utils/filter_scp.pl -f 2 ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt ${unreliable_data_dir}/segments |" ark:- \| \
####    segmentation-combine-segments-to-recordings ark:- "ark,t:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt ${unreliable_data_dir}/reco2utt |" \
####    ark:- \| \
####    segmentation-create-subsegments --filter-label=1 --subsegment-label=0 --ignore-missing \
####    "ark:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt $corrupted_data_dir/utt2num_frames | segmentation-init-from-lengths ark,t:- ark:- |" \
####    ark:- ark:- \| \
####    segmentation-intersect-segments --mismatch-label=0 \
####    "ark:utils/filter_scp.pl ${overlap_data_dir}/split${reco_nj}reco/JOB/reco2utt $corrupted_data_dir/sad_seg.scp | segmentation-post-process --remove-labels=0:2:3 scp:- ark:- |" \
####    ark:- ark:- \| \
####    segmentation-post-process --remove-labels=0 ark:- ark:- \| \
####    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
####    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
####    ark,scp:$overlap_labels_dir/deriv_weights_for_overlapped_speech.JOB.ark,$overlap_labels_dir/deriv_weights_for_overlapped_speech.JOB.scp
####
####  for n in `seq $reco_nj`; do
####    cat $overlap_labels_dir/deriv_weights_for_overlapped_speech.${n}.scp
####  done > $corrupted_data_dir/deriv_weights_for_overlapped_speech.scp
####fi
####
####exit 0
