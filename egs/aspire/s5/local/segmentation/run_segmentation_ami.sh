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
extra_left_context=100 
extra_right_context=20
task=SAD
iter=final

segmentation_stage=-1
sil_prior=0.7
speech_prior=0.3
min_silence_duration=30
min_speech_duration=10
frame_subsampling_factor=3

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

cat $src_dir/data/sdm1/dev_ihmdata/reco2utt | \
  awk 'BEGIN{i=1} {print $1" "i; i++;}' > \
  $src_dir/data/sdm1/dev_ihmdata/reco.txt

if [ $stage -le 5 ]; then
  utils/data/get_reco2num_frames.sh --frame-shift 0.01 --frame-overlap 0.015 \
    --cmd queue.pl --nj 18 \
    $src_dir/data/sdm1/dev

  # Get a filter that selects only regions within the manual segments.
  $train_cmd $dir/log/get_manual_segments_regions.log \
    segmentation-init-from-segments --shift-to-zero=false $src_dir/data/sdm1/dev/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- ark,t:$src_dir/data/sdm1/dev/reco2utt ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=1 \
    "ark:segmentation-init-from-lengths --label=0 ark,t:$src_dir/data/sdm1/dev/reco2num_frames ark:- |" ark:- ark,t:- \| \
    perl -ane '$F[3] = 10000; $F[$#F-1] = 10000; print join(" ", @F) . "\n";' \| \
    segmentation-post-process --merge-labels=0:1 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-post-process --merge-labels=10000 --merge-dst-label=0 --merge-adjacent-segments \
    --max-intersegment-length=10000 ark,t:- \
    "ark:| gzip -c > $dir/manual_segments_regions.seg.gz" 
fi

if [ $stage -le 6 ]; then
  # Reference RTTM where SPEECH frames are obtainted by combining IHM VAD alignments
  $train_cmd $dir/log/get_ref_spk_seg.log \
    segmentation-combine-segments scp:$dir/sad_seg.scp \
    "ark:segmentation-init-from-segments --shift-to-zero=false $src_dir/data/sdm1/dev_ihmdata/segments ark:- |" \
    ark,t:$src_dir/data/sdm1/dev_ihmdata/reco2utt ark:- \| \
    segmentation-copy --keep-label=1 ark:- ark:- \| \
    segmentation-copy --utt2label-rspecifier=ark,t:$src_dir/data/sdm1/dev_ihmdata/reco.txt \
    ark:- ark:- \| \
    segmentation-merge-recordings \
    "ark,t:utils/utt2spk_to_spk2utt.pl $src_dir/data/sdm1/dev_ihmdata/ihm2sdm_reco |" \
    ark:- "ark:| gzip -c > $dir/ref_spk_seg.gz"
fi

if [ $stage -le 7 ]; then
  # To get the actual RTTM, we need to add no-score
  $train_cmd $dir/log/get_ref_rttm.log \
    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/dev/reco2num_frames \
    "ark:gunzip -c $dir/ref_spk_seg.gz | segmentation-post-process --remove-labels=0 ark:- ark:- |" \
    ark:/dev/null ark:- \| \
    segmentation-init-from-ali ark:- ark:- \| \
    segmentation-post-process --merge-labels=1:2:3:4:5:6:7:8:9:10 --merge-dst-label=1 \
    --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-create-subsegments --filter-label=0 --subsegment-label=10000 \
    ark:- "ark:gunzip -c $dir/manual_segments_regions.seg.gz |" ark:- \| \
    segmentation-post-process --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel \
    --no-score-label=10000 ark:- $dir/ref.rttm

  # Get RTTM for overlapped speech detection with 3 classes
  # 0 -> SILENCE, 1 -> SINGLE_SPEAKER, 2 -> OVERLAP
  $train_cmd $dir/log/get_overlapping_rttm.log \
    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/dev/reco2num_frames \
    "ark:gunzip -c $dir/ref_spk_seg.gz | segmentation-post-process --remove-labels=0 ark:- ark:- |" \
    ark:/dev/null ark:- \| \
    segmentation-init-from-ali ark:- ark:- \| \
    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=2 \
    --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-create-subsegments --filter-label=0 --subsegment-label=10000 \
    ark:- "ark:gunzip -c $dir/manual_segments_regions.seg.gz |" ark:- \| \
    segmentation-post-process --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-to-rttm --map-to-speech-and-sil=false --reco2file-and-channel=$dir/reco2file_and_channel \
    --no-score-label=10000 ark:- $dir/overlapping_speech_ref.rttm
fi

if [ $stage -le 8 ]; then
  # Get a filter that selects only regions of speech 
  $train_cmd $dir/log/get_speech_filter.log \
    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/dev/reco2num_frames \
    "ark:gunzip -c $dir/ref_spk_seg.gz | segmentation-post-process --remove-labels=0 ark:- ark:- |" \
    ark:/dev/null ark:- \| \
    segmentation-init-from-ali ark:- ark:- \| \
    segmentation-post-process --merge-labels=1:2:3:4:5:6:7:8:9:10 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-create-subsegments --filter-label=0 --subsegment-label=0 \
    ark:- "ark:gunzip -c $dir/manual_segments_regions.seg.gz |" ark:- \| \
    segmentation-post-process --merge-adjacent-segments --max-intersegment-length=10000 \
    ark:- "ark:| gzip -c > $dir/manual_segments_speech_regions.seg.gz"
fi
  
hyp_dir=${nnet_dir}/segmentation_ami_sdm1_dev_whole_bp/ami_sdm1_dev

if [ $stage -le 9 ]; then
  steps/segmentation/do_segmentation_data_dir.sh --reco-nj 18 \
    --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp --do-downsampling true \
    --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
    --output-name output-speech --frame-subsampling-factor $frame_subsampling_factor --iter $iter \
    --stage $segmentation_stage \
    $src_dir/data/sdm1/dev $nnet_dir mfcc_hires_bp $hyp_dir
fi

sad_dir=${nnet_dir}/sad_ami_sdm1_dev_whole_bp/
hyp_dir=${hyp_dir}_seg

if [ $stage -le 10 ]; then
  utils/data/get_reco2utt.sh $src_dir/data/sdm1/dev_ihmdata
  utils/data/get_reco2utt.sh $hyp_dir

  segmentation-init-from-segments --shift-to-zero=false $hyp_dir/segments ark:- | \
    segmentation-combine-segments-to-recordings ark:- ark,t:$hyp_dir/reco2utt ark:- | \
    segmentation-to-ali --length-tolerance=48 --lengths-rspecifier=ark,t:$src_dir/data/sdm1/dev/reco2num_frames \
    ark:- ark:- | \
    segmentation-init-from-ali ark:- ark:- | \
    segmentation-to-rttm --reco2file-and-channel=$dir/reco2file_and_channel ark:- $hyp_dir/sys.rttm

  #steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
  #  $hyp_dir/utt2spk \
  #  $hyp_dir/segments \
  #  $dir/reco2file_and_channel \
  #  /dev/stdout | spkr2sad.pl > $hyp_dir/sys.rttm
fi

if [ $stage -le 11 ]; then
  cat <<EOF
md-eval.pl -s <(cat $hyp_dir/sys.rttm | rttmSmooth.pl -s 0) \
  -r <(cat $dir/ref.rttm | rttmSmooth.pl -s 0 ) \
  -u $dir/uem -c 0.25
EOF

  md-eval.pl -s <(cat $hyp_dir/sys.rttm | rttmSmooth.pl -s 0) \
    -r <(cat $dir/ref.rttm | rttmSmooth.pl -s 0 ) \
    -u $dir/uem -c 0.25
fi

if [ $task == "OVLP" ] || [ $task == "OVLP_SAD" ]; then
  hyp_dir=${nnet_dir}/segmentation_ovlp_ami_sdm1_dev_whole_bp/ami_sdm1_dev

  if [ $task == "OVLP" ]; then
    if [ $stage -le 12 ]; then
      steps/segmentation/do_segmentation_data_dir.sh --reco-nj 18 \
        --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp --do-downsampling true \
        --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
        --segmentation-config conf/segmentation_ovlp.conf \
        --output-name output-overlapped_speech \
        --sil-prior $sil_prior --speech-prior $speech_prior \
        --min-silence-duration $min_silence_duration --min-speech-duration $min_speech_duration \
        --sad-name ovlp --segmentation-name segmentation_ovlp \
        --frame-subsampling-factor $frame_subsampling_factor --iter $iter \
        --stage $segmentation_stage \
        $src_dir/data/sdm1/dev $nnet_dir mfcc_hires_bp $hyp_dir
    fi

    sad_dir=${nnet_dir}/sad_ami_sdm1_dev_whole_bp/
    ovlp_dir=${nnet_dir}/ovlp_ami_sdm1_dev_whole_bp/

    likes_dir=${nnet_dir}/sad_ovlp_ami_sdm1_dev_whole_bp/
    
    if [ $stage -le 13 ]; then
      $train_cmd JOB=1:18 $likes_dir/log/get_sad_ovlp_likes.JOB.log \
        paste-feats "ark:gunzip -c $sad_dir/log_likes.JOB.gz | select-feats 1 ark:- ark:- |" \
        "ark:gunzip -c $sad_dir/log_likes.JOB.gz | select-feats 1 ark:- ark:- |" ark:- \| \
        matrix-sum "ark:gunzip -c $ovlp_dir/log_likes.JOB.gz |" ark:- ark:- \| \
        paste-feats "ark:gunzip -c $sad_dir/log_likes.JOB.gz | select-feats 0 ark:- ark:- |" \
        ark:- "ark:| gzip -c > $likes_dir/log_likes.JOB.gz"
      cp $sad_dir/num_jobs $likes_dir
    fi
  else
    if [ $stage -le 12 ]; then
      steps/segmentation/do_segmentation_data_dir_generic.sh --reco-nj 18 \
        --mfcc-config conf/mfcc_hires_bp.conf --feat-affix bp --do-downsampling true \
        --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
        --segmentation-config conf/segmentation_ovlp.conf \
        --output-name output-overlapping_sad \
        --min-durations 30:10:10 --priors 0.5:0.35:0.15 \
        --sad-name ovlp_sad --segmentation-name segmentation_ovlp_sad \
        --frame-subsampling-factor $frame_subsampling_factor --iter $iter \
        --stage $segmentation_stage \
        $src_dir/data/sdm1/dev $nnet_dir mfcc_hires_bp $hyp_dir
    fi

    likes_dir=${nnet_dir}/ovlp_sad_ami_sdm1_dev_whole_bp/
  fi

  hyp_dir=${hyp_dir}_seg
  mkdir -p $hyp_dir

  seg_dir=${nnet_dir}/segmentation_ovlp_sad_ami_sdm1_dev_whole_bp/
  lang=${seg_dir}/lang

  if [ $stage -le 14 ]; then
  mkdir -p $lang
  steps/segmentation/internal/prepare_sad_lang.py \
    --phone-transition-parameters="--phone-list=1 --min-duration=10 --end-transition-probability=0.1" \
    --phone-transition-parameters="--phone-list=2 --min-duration=3 --end-transition-probability=0.1" \
    --phone-transition-parameters="--phone-list=3 --min-duration=3 --end-transition-probability=0.1" $lang
  cp $lang/phones.txt $lang/words.txt
  
  feat_dim=2    # dummy. We don't need this.
  $train_cmd $seg_dir/log/create_transition_model.log gmm-init-mono \
    $lang/topo $feat_dim - $seg_dir/tree \| \
    copy-transition-model --binary=false - $seg_dir/trans.mdl || exit 1
fi

  if [ $stage -le 15 ]; then
  
  cat > $lang/word2prior <<EOF
1 0.01
2 0.01
3 0.98
EOF
  
  steps/segmentation/internal/make_G_fst.py --word2prior-map $lang/word2prior | \
    fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false \
    > $lang/G.fst
fi

  if [ $stage -le 16 ]; then
    $train_cmd $seg_dir/log/make_vad_graph.log \
      steps/segmentation/internal/make_sad_graph.sh --iter trans \
      $lang $seg_dir $seg_dir/graph_test || exit 1
  fi

  if [ $stage -le 17 ]; then
    steps/segmentation/decode_sad.sh \
      --acwt 1 --beam 10 --max-active 7000 \
      $seg_dir/graph_test $likes_dir $seg_dir
  fi

  if [ $stage -le 18 ]; then
    cat <<EOF > $hyp_dir/labels_map
1 0
2 1
3 2
EOF
    gunzip -c $seg_dir/ali.*.gz | \
      segmentation-init-from-ali ark:- ark:- | \
      segmentation-copy --frame-subsampling-factor=$frame_subsampling_factor \
      --label-map=$hyp_dir/labels_map ark:- ark:- | \
      segmentation-to-rttm --map-to-speech-and-sil=false \
      --reco2file-and-channel=$dir/reco2file_and_channel ark:- $hyp_dir/sys.rttm
  fi
  # Get RTTM for overlapped speech detection with 3 classes
  # 0 -> SILENCE, 1 -> SINGLE_SPEAKER, 2 -> OVERLAP
  $train_cmd $dir/log/get_overlapping_rttm.log \
    segmentation-get-stats --lengths-rspecifier=ark,t:$src_dir/data/sdm1/dev/reco2num_frames \
    "ark:gunzip -c $dir/ref_spk_seg.gz | segmentation-post-process --remove-labels=0 ark:- ark:- |" \
    ark:/dev/null ark:- \| \
    segmentation-init-from-ali ark:- ark:- \| \
    segmentation-post-process --merge-labels=2:3:4:5:6:7:8:9:10 --merge-dst-label=2 ark:- ark:- \| \
    segmentation-create-subsegments --filter-label=0 --subsegment-label=10000 \
    ark:- "ark:gunzip -c $dir/manual_segments_regions.seg.gz |" ark:- \| \
    segmentation-post-process --merge-adjacent-segments --max-intersegment-length=10000 ark:- ark:- \| \
    segmentation-to-rttm --map-to-speech-and-sil=false --reco2file-and-channel=$dir/reco2file_and_channel \
    --no-score-label=10000 ark:- $dir/overlapping_speech_ref.rttm

  if [ $stage -le 19 ]; then
    cat <<EOF
md-eval.pl -s <(cat $hyp_dir/sys.rttm | rttmSmooth.pl -s 0) \
  -r <(cat $dir/overlapping_speech_ref.rttm | rttmSmooth.pl -s 0) \
  -u $dir/uem
EOF

    md-eval.pl -s <(cat $hyp_dir/sys.rttm | rttmSmooth.pl -s 0) \
      -r <(cat $dir/overlapping_speech_ref.rttm | rttmSmooth.pl -s 0) \
      -u $dir/uem
  fi
else
  echo "$0: Unknown task $task"
  exit 1
fi

#md-eval.pl -s <( segmentation-init-from-segments --shift-to-zero=false exp/nnet3_sad_snr/nnet_tdnn_j_n4/segmentation_ami_sdm1_dev_whole_bp/ami_sdm1_dev_seg/segments ark:- | segmentation-combine-segments-to-recordings ark:- ark,t:exp/nnet3_sad_snr/nnet_tdnn_j_n4/segmentation_ami_sdm1_dev_whole_bp/ami_sdm1_dev_seg/reco2utt ark:- | segmentation-to-ali --length-tolerance=1000 --lengths-rspecifier=ark,t:data/ami_sdm1_dev_whole_bp_hires/utt2num_frames ark:- ark:- |
#segmentation-init-from-ali ark:- ark:- | segmentation-to-rttm ark:- - | grep SPEECH | rttmSmooth.pl -s 0)
