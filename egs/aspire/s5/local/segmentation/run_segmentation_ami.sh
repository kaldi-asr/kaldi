#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

. cmd.sh
. path.sh

set -e 
set -o pipefail
set -u

stage=-1
nnet_dir=exp/nnet3_sad_snr/nnet_tdnn_k_n4

. utils/parse_options.sh

export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH

src_dir=/export/a09/vmanoha1/workspace_asr_diarization/egs/ami/s5b # AMI src_dir
dir=exp/sad_ami_sdm1_dev/ref

mkdir -p $dir

# Expecting user to have done run.sh to run the AMI recipe in $src_dir for
# both sdm and ihm microphone conditions

if [ $stage -le 1 ]; then
  ( 
  cd $src_dir
  local/prepare_parallel_train_data.sh --train-set dev sdm1

  awk '{print $1" "$2}' $src_dir/data/ihm/dev/segments > \
    $src_dir/data/ihm/dev/utt2reco
  awk '{print $1" "$2}' $src_dir/data/sdm1/dev/segments > \
    $src_dir/data/sdm1/dev/utt2reco

  cat $src_dir/data/sdm1/dev_ihmdata/ihmutt2utt | \
    utils/apply_map.pl -f 1 $src_dir/data/ihm/dev/utt2reco | \
    utils/apply_map.pl -f 2 $src_dir/data/sdm1/dev/utt2reco | \
    sort -u > $src_dir/data/sdm1/dev_ihmdata/ihm2sdm_reco
  )
fi

if [ $stage -le 2 ]; then
  (
  cd $src_dir
  utils/data/get_reco2utt.sh $src_dir/data/sdm1/dev
  )

  phone_map=$dir/phone_map
  steps/segmentation/get_sad_map.py \
    $src_dir/data/lang | utils/sym2int.pl -f 1 $src_dir/data/lang/phones.txt > \
    $phone_map
fi

if [ $stage -le 3 ]; then
  # Expecting user to have run local/run_cleanup_segmentation.sh in $src_dir
  (
  cd $src_dir
  steps/align_fmllr.sh --nj 18 --cmd "$train_cmd" \
    data/sdm1/dev_ihmdata data/lang \
    exp/ihm/tri3_cleaned \
    exp/sdm1/tri3_cleaned_dev_ihmdata
  )
fi

if [ $stage -le 4 ]; then
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$train_cmd" \
    $src_dir/exp/sdm1/tri3_cleaned_dev_ihmdata $phone_map $dir
fi

echo "A 1" > $dir/channel_map
cat $src_dir/data/sdm1/dev/reco2file_and_channel | \
  utils/apply_map.pl -f 3 $dir/channel_map > $dir/reco2file_and_channel

if [ $stage -le 5 ]; then
  $train_cmd $dir/log/get_ref_rttm.log \
    segmentation-combine-segments scp:$dir/sad_seg.scp \
    "ark:segmentation-init-from-segments --shift-to-zero=false $src_dir/data/sdm1/dev_ihmdata/segments ark:- |" \
    ark,t:$src_dir/data/sdm1/dev_ihmdata/reco2utt ark:- \| \
    segmentation-merge-recordings \
    "ark,t:utils/utt2spk_to_spk2utt.pl $src_dir/data/sdm1/dev_ihmdata/ihm2sdm_reco |" \
    ark:- ark:- \| \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel \
    ark:- $dir/ref.rttm
fi
  
if [ $stage -le 6 ]; then
  $train_cmd $dir/log/get_uem.log \
    segmentation-init-from-segments --shift-to-zero=false $src_dir/data/sdm1/dev/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- ark,t:$src_dir/data/sdm1/dev/reco2utt ark:- \| \
    segmentation-post-process --remove-labels=0 --merge-adjacent-segments \
    --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel \
    ark:- - \| grep SPEECH \| grep SPEAKER \| \
    rttmSmooth.pl -s 0 \| awk '{ print $2" "$3" "$4" "$5+$4 }' '>' $dir/uem
fi

hyp_dir=$nnet_dir/segmentation_ami_sdm1_dev_whole_bp

if [ $stage -le 7 ]; then
  steps/segmentation/do_segmentation_data_dir.sh --reco-nj 18 \
    --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp --do-downsampling true \
    --extra-left-context 100 --extra-right-context 20 \
    --output-name output-speech --frame-subsampling-factor 6 \
    $src_dir/data/sdm1/dev data/ami_sdm1_dev $nnet_dir
fi


if [ $stage -le 8 ]; then
  utils/data/get_reco2utt.sh $src_dir/data/sdm1/dev_ihmdata
  
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
    $hyp_dir/ami_sdm1_dev_seg/utt2spk \
    $hyp_dir/ami_sdm1_dev_seg/segments \
    $dir/reco2file_and_channel \
    /dev/stdout | spkr2sad.pl > $hyp_dir/sys.rttm
fi

if [ $stage -le 9 ]; then
  md-eval.pl -s <(cat $hyp_dir/sys.rttm | grep speech | rttmSmooth.pl -s 0) \
    -r <(cat $dir/ref.rttm | grep SPEECH | rttmSmooth.pl -s 0 ) \
    -u $dir/uem -c 0.25
fi

#md-eval.pl -s <( segmentation-init-from-segments --shift-to-zero=false exp/nnet3_sad_snr/nnet_tdnn_j_n4/segmentation_ami_sdm1_dev_whole_bp/ami_sdm1_dev_seg/segments ark:- | segmentation-combine-segments-to-recordings ark:- ark,t:exp/nnet3_sad_snr/nnet_tdnn_j_n4/segmentation_ami_sdm1_dev_whole_bp/ami_sdm1_dev_seg/reco2utt ark:- | segmentation-to-ali --length-tolerance=1000 --lengths-rspecifier=ark,t:data/ami_sdm1_dev_whole_bp_hires/utt2num_frames ark:- ark:- |
#segmentation-init-from-ali ark:- ark:- | segmentation-to-rttm ark:- - | grep SPEECH | rttmSmooth.pl -s 0)
