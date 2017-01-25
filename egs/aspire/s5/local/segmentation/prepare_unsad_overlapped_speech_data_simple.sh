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
snr_db_threshold=10
stage=-1

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <corrupted-data-dir> <orig-corrupted-data-dir> <utt-vad-dir> <temp-dir> <out-labels-dir>"
  echo " e.g.: $0 data/fisher_train_100k_sp_75k_seg_ovlp_corrupted_hires_bp data/fisher_train_100k_sp_75k_seg_ovlp_corrupted exp/unsad/make_unsad_fisher_train_100k/tri4a_ali_fisher_train_100k_sp_vad_fisher_train_100k_sp exp/unsad overlapping_sad_labels"
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

overlap_dir=$tmpdir/make_overlapping_sad_labels_${corrupted_data_id}

# make $overlap_labels_dir an absolute pathname.
overlap_labels_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $overlap_labels_dir ${PWD}`
mkdir -p $overlap_labels_dir

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

  # 1) segmentation-init-from-additive-signals-info converts the informtation 
  # written out but by steps/data/make_corrupted_data_dir.py in overlapped_segments_info.txt
  # and converts it to segments. It then adds those segments to the 
  # segments already present ($corrupted_data_dir/sad_seg.scp)
  # 2) Retain only the speech segments (label 1) from these.
  # 3) Convert this to overlap stats using segmentation-get-stats, which 
  # writes for each frame the number of overlapping segments.
  # 4) Convert this per-frame "alignment" information to segmentation 
  # ($overlap_dir/overlap_seg.*.gz).
  $cmd JOB=1:$nj $overlap_dir/log/get_overlapping_sad_seg.JOB.log \
    segmentation-init-from-additive-signals-info --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    --junk-label=10000 \
    --additive-signals-segmentation-rspecifier=scp:$utt_vad_dir/sad_seg.scp \
    "ark,t:utils/filter_scp.pl ${orig_corrupted_data_dir}/split${reco_nj}reco/JOB/reco2utt $orig_corrupted_data_dir/overlapped_segments_info.txt |" \
    ark:- \| \
    segmentation-merge "scp:utils/filter_scp.pl ${corrupted_data_dir}/split${nj}/JOB/utt2spk $corrupted_data_dir/sad_seg.scp |" ark:- ark:- \| \
    segmentation-get-stats --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    ark:- ark:/dev/null ark:/dev/null ark:- \| \
    classes-per-frame-to-labels --junk-label=10000 ark:- ark:- \| \
    segmentation-init-from-ali ark:- \
    "ark:| gzip -c > $overlap_dir/overlap_sad_seg.JOB.gz"
fi

if [ $stage -le 2 ]; then
  # Call labels >2, i.e. regions where more than 1 speaker overlap as overlapping speech. labels = 1 is single speaker and labels = 0 is silence.
  # Write this out in alignment format as "overlapping_sad_labels"
  $cmd JOB=1:$nj $overlap_dir/log/get_overlapping_sad_labels.JOB.log \
    gunzip -c $overlap_dir/overlap_sad_seg.JOB.gz \| \
    segmentation-post-process --remove-labels=10000 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- \
    ark,scp:$overlap_labels_dir/overlapping_sad_labels_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/overlapping_sad_labels_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/overlapping_sad_labels_${corrupted_data_id}.$n.scp
  done > ${corrupted_data_dir}/overlapping_sad_labels.scp
fi

if [ $stage -le 3 ]; then
  # Find regions where there is at least one speaker speaking.
  $cmd JOB=1:$nj $overlap_dir/log/get_speech_feat.JOB.log \
    gunzip -c $overlap_dir/overlap_sad_seg.JOB.gz \| \
    segmentation-post-process --remove-labels=10000 ark:- ark:- \| \
    segmentation-post-process --merge-labels=1:2 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| \
    vector-to-feat ark:- \
    ark,scp:$overlap_labels_dir/speech_feat_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/speech_feat_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/speech_feat_${corrupted_data_id}.$n.scp
  done > ${corrupted_data_dir}/speech_feat.scp
fi

if [ $stage -le 4 ]; then
  # Deriv weights is 1 everywhere but the 
  # unreliable regions.
  $cmd JOB=1:$nj $overlap_dir/log/get_deriv_weights.JOB.log \
    gunzip -c $overlap_dir/overlap_sad_seg.JOB.gz \| \
    segmentation-post-process --merge-labels=0:1:2 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-post-process --merge-labels=10000 --merge-dst-label=0 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${corrupted_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
    ark,scp:$overlap_labels_dir/deriv_weights_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/deriv_weights_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/deriv_weights_$corrupted_data_id.${n}.scp
  done > $corrupted_data_dir/deriv_weights.scp
fi
 
snr_threshold=`perl -e "print $snr_db_threshold / 10.0 * log(10.0)"`

cat <<EOF > $overlap_dir/invert_labels.map
0 2
1 1
EOF

if [ $stage -le 5 ]; then
  if [ ! -f $corrupted_data_dir/log_snr.scp ]; then
    echo "$0: Could not find $corrupted_data_dir/log_snr.scp. Run local/segmentation/do_corruption_data_dir_overlapped_speech.sh."
    exit 1
  fi

  $cmd JOB=1:$nj $overlap_dir/log/fix_overlapping_sad_labels.JOB.log \
    copy-matrix --apply-power=1 \
    "scp:utils/filter_scp.pl $corrupted_data_dir/split$nj/JOB/utt2spk $corrupted_data_dir/log_snr.scp |" \
    ark:- \| extract-column ark:- ark,t:- \| \
    steps/segmentation/quantize_vector.pl $snr_threshold \| \
    segmentation-init-from-ali ark,t:- ark:- \| \
    segmentation-copy --label-map=$overlap_dir/invert_labels.map ark:- ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=1 \
    "ark:utils/filter_scp.pl $corrupted_data_dir/split$nj/JOB/utt2spk $corrupted_data_dir/overlapping_sad_labels.scp | segmentation-init-from-ali scp:- ark:- |" ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:$corrupted_data_dir/utt2num_frames \
    ark:- ark,scp:$overlap_labels_dir/overlapping_sad_labels_fixed_${corrupted_data_id}.JOB.ark,$overlap_labels_dir/overlapping_sad_labels_fixed_${corrupted_data_id}.JOB.scp

  for n in `seq $nj`; do
    cat $overlap_labels_dir/overlapping_sad_labels_fixed_${corrupted_data_id}.$n.scp
  done > $corrupted_data_dir/overlapping_sad_labels_fixed.scp
fi

exit 0
