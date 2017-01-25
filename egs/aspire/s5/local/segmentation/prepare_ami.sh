#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

. cmd.sh
. path.sh

set -e 
set -o pipefail
set -u

stage=-1

dataset=dev
nj=18

. utils/parse_options.sh

export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH

src_dir=/export/a09/vmanoha1/workspace_asr_diarization/egs/ami/s5b # AMI src_dir
dir=exp/sad_ami_sdm1_${dataset}/ref

mkdir -p $dir

# Expecting user to have done run.sh to run the AMI recipe in $src_dir for
# both sdm and ihm microphone conditions

if [ $stage -le 1 ]; then
  ( 
  cd $src_dir
  local/prepare_parallel_train_data.sh --train-set ${dataset} sdm1

  awk '{print $1" "$2}' $src_dir/data/ihm/${dataset}/segments > \
    $src_dir/data/ihm/${dataset}/utt2reco
  awk '{print $1" "$2}' $src_dir/data/sdm1/${dataset}/segments > \
    $src_dir/data/sdm1/${dataset}/utt2reco

  cat $src_dir/data/sdm1/${dataset}_ihmdata/ihmutt2utt | \
    utils/filter_scp.pl -f 1 $src_dir/data/ihm/${dataset}/utt2reco | \
    utils/apply_map.pl -f 1 $src_dir/data/ihm/${dataset}/utt2reco | \
    utils/filter_scp.pl -f 2 $src_dir/data/sdm1/${dataset}/utt2reco | \
    utils/apply_map.pl -f 2 $src_dir/data/sdm1/${dataset}/utt2reco | \
    sort -u > $src_dir/data/sdm1/${dataset}_ihmdata/ihm2sdm_reco
  )
fi

[ ! -s $src_dir/data/sdm1/${dataset}_ihmdata/ihm2sdm_reco ] && echo "Empty $src_dir/data/sdm1/${dataset}_ihmdata/ihm2sdm_reco!" && exit 1

phone_map=$dir/phone_map
if [ $stage -le 2 ]; then
  (
  cd $src_dir
  utils/data/get_reco2utt.sh $src_dir/data/sdm1/${dataset}
  
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
    data/sdm1/${dataset}_ihmdata exp/sdm1/make_mfcc mfcc_sdm1
  steps/compute_cmvn_stats.sh \
    data/sdm1/${dataset}_ihmdata exp/sdm1/make_mfcc mfcc_sdm1
  utils/fix_data_dir.sh data/sdm1/${dataset}_ihmdata
  )

  steps/segmentation/get_sad_map.py \
    $src_dir/data/lang | utils/sym2int.pl -f 1 $src_dir/data/lang/phones.txt > \
    $phone_map
fi

if [ $stage -le 3 ]; then
  # Expecting user to have run local/run_cleanup_segmentation.sh in $src_dir
  (
  cd $src_dir
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/sdm1/${dataset}_ihmdata data/lang \
    exp/ihm/tri3_cleaned \
    exp/sdm1/tri3_cleaned_${dataset}_ihmdata
  )
fi

if [ $stage -le 4 ]; then
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$train_cmd" \
    $src_dir/exp/sdm1/tri3_cleaned_${dataset}_ihmdata $phone_map $dir
fi

echo "A 1" > $dir/channel_map
cat $src_dir/data/sdm1/${dataset}/reco2file_and_channel | \
  utils/apply_map.pl -f 3 $dir/channel_map > $dir/reco2file_and_channel

utils/data/get_reco2utt.sh $src_dir/data/sdm1/${dataset}_ihmdata
cat $src_dir/data/sdm1/${dataset}_ihmdata/reco2utt | \
  awk 'BEGIN{i=1} {print $1" "i; i++;}' > \
  $src_dir/data/sdm1/${dataset}_ihmdata/reco.txt

if [ $stage -le 5 ]; then
  # Reference RTTM where SPEECH frames are obtainted by combining IHM VAD alignments
  cat $src_dir/data/sdm1/${dataset}_ihmdata/reco.txt | \
    awk '{print $1" 1:"$2" 10000:10000 0:0"}' > $dir/ref_spk2label_map

  $train_cmd $dir/log/get_ref_spk_seg.log \
    segmentation-combine-segments --include-missing-utt-level-segmentations scp:$dir/sad_seg.scp \
    "ark:segmentation-init-from-segments --segment-label=10000 --shift-to-zero=false $src_dir/data/sdm1/${dataset}_ihmdata/segments ark:- |" \
    ark,t:$src_dir/data/sdm1/${dataset}_ihmdata/reco2utt ark:- \| \
    segmentation-copy --utt2label-map-rspecifier=ark,t:$dir/ref_spk2label_map \
    ark:- ark:- \| \
    segmentation-merge-recordings \
    "ark,t:utils/utt2spk_to_spk2utt.pl $src_dir/data/sdm1/${dataset}_ihmdata/ihm2sdm_reco |" \
    ark:- "ark:| gzip -c > $dir/ref_spk_seg.gz"
fi

if [ $stage -le 6 ]; then
  utils/data/get_reco2num_frames.sh --frame-shift 0.01 --frame-overlap 0.015 \
    --cmd queue.pl --nj $nj \
    $src_dir/data/sdm1/${dataset}

  ## Get a filter that selects only regions within the manual segments.
  #$train_cmd $dir/log/get_manual_segments_regions.log \
  #  segmentation-init-from-segments --shift-to-zero=false $src_dir/data/sdm1/${dataset}/segments ark:- \| \
  #  segmentation-combine-segments-to-recordings ark:- ark,t:$src_dir/data/sdm1/${dataset}/reco2utt ark:- \| \
  #  segmentation-create-subsegments --filter-label=1 --subsegment-label=1 \
  #  "ark:segmentation-init-from-lengths --label=0 ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames ark:- |" ark:- ark,t:- \| \
  #  perl -ane '$F[3] = 10000; $F[$#F-1] = 10000; print join(" ", @F) . "\n";' \| \
  #  segmentation-create-subsegments --filter-label=10000 --subsegment-label=10000 \
  #  ark,t:- "ark:gunzip -c $dir/ref_spk_seg.gz |" ark:- \| \
  #  segmentation-post-process --merge-labels=0:1 --merge-dst-label=1 ark:- ark:- \| \
  #  segmentation-post-process --merge-labels=10000 --merge-dst-label=0 --merge-adjacent-segments \
  #  --max-intersegment-length=10000 ark,t:- \
  #  "ark:| gzip -c > $dir/manual_segments_regions.seg.gz" 
fi

if [ $stage -le 7 ]; then
  $train_cmd $dir/log/get_overlap_sad_seg.log \
    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames \
    "ark:gunzip -c $dir/ref_spk_seg.gz |" \
    ark:/dev/null ark:/dev/null ark:- \| \
    classes-per-frame-to-labels --junk-label=10000 ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    "ark:| gzip -c > $dir/overlap_sad_seg.gz"
fi

if [ $stage -le 8 ]; then
  # To get the actual RTTM, we need to add no-score
  $train_cmd $dir/log/get_ref_rttm.log \
    gunzip -c $dir/overlap_sad_seg.gz \| \
    segmentation-post-process --merge-labels=1:2 --merge-dst-label=1 \
    ark:- ark:- \| \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel \
    --no-score-label=10000 ark:- $dir/ref.rttm
  
  # Get RTTM for overlapped speech detection with 3 classes
  # 0 -> SILENCE, 1 -> SINGLE_SPEAKER, 2 -> OVERLAP
  $train_cmd $dir/log/get_ref_rttm.log \
    gunzip -c $dir/overlap_sad_seg.gz \| \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel \
    --no-score-label=10000 --map-to-speech-and-sil=false ark:- $dir/overlapping_speech_ref.rttm
fi


#if [ $stage -le 8 ]; then
#  # Get RTTM for overlapped speech detection with 3 classes
#  # 0 -> SILENCE, 1 -> SINGLE_SPEAKER, 2 -> OVERLAP
#  $train_cmd $dir/log/get_overlapping_rttm.log \
#    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames \
#    "ark:gunzip -c $dir/ref_spk_seg.gz | segmentation-post-process --remove-labels=0:10000 ark:- ark:- |" \
#    ark:/dev/null ark:- \| \
#    segmentation-init-from-ali ark:- ark:- \| \
#    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=2 \
#    --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
#    segmentation-create-subsegments --filter-label=0 --subsegment-label=10000 \
#    ark:- "ark:gunzip -c $dir/manual_segments_regions.seg.gz |" ark:- \| \
#    segmentation-post-process --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
#    segmentation-to-rttm --map-to-speech-and-sil=false --reco2file-and-channel=$dir/reco2file_and_channel \
#    --no-score-label=10000 ark:- $dir/overlapping_speech_ref.rttm
#fi

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

if [ $stage -le 9 ]; then
  # Get a filter that selects only regions of speech 
  $train_cmd $dir/log/get_speech_filter.log \
    gunzip -c $dir/overlap_sad_seg.gz \| \
    segmentation-post-process --merge-labels=1:2 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-post-process --remove-labels=10000 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames \
    ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| \
    copy-vector ark,t: ark,scp:$dir/deriv_weights_for_overlapping_sad.ark,$dir/deriv_weights_for_overlapping_sad.scp
  
  # Get deriv weights
  $train_cmd $dir/log/get_speech_filter.log \
    gunzip -c $dir/overlap_sad_seg.gz \| \
    segmentation-post-process --merge-labels=0:1:2 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-post-process --remove-labels=10000 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames \
    ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| \
    copy-vector ark,t: ark,scp:$dir/deriv_weights.ark,$dir/deriv_weights.scp
fi

if [ $stage -le 10 ]; then
  $train_cmd $dir/log/get_overlapping_sad.log \
    gunzip -c $dir/overlap_sad_seg.gz \| \
    segmentation-post-process --remove-labels=10000 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:$src_dir/data/sdm1/${dataset}/reco2num_frames \
    ark:- ark,scp:$dir/overlapping_sad_labels.ark,$dir/overlapping_sad_labels.scp
fi

if false && [ $stage -le 11 ]; then
  utils/data/convert_data_dir_to_whole.sh \
    $src_dir/data/sdm1/${dataset} data/ami_sdm1_${dataset}_whole
  utils/fix_data_dir.sh \
    data/ami_sdm1_${dataset}_whole
  utils/copy_data_dir.sh \
    data/ami_sdm1_${dataset}_whole data/ami_sdm1_${dataset}_whole_hires_bp
  utils/data/downsample_data_dir.sh 8000 data/ami_sdm1_${dataset}_whole_hires_bp

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf --nj $nj \
    data/ami_sdm1_${dataset}_whole_hires_bp exp/make_hires_bp mfcc_hires_bp
  steps/compute_cmvn_stats.sh --fake \
    data/ami_sdm1_${dataset}_whole_hires_bp exp/make_hires_bp mfcc_hires_bp
  utils/fix_data_dir.sh \
    data/ami_sdm1_${dataset}_whole_hires_bp
fi
