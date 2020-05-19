#!/usr/bin/env bash

# Copyright 2018  Johns Hopkins University (author: Daniel Povey)
#           2018  Mahsa Yarmohammadi
#           2018  Yiming Wang


# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

language=swahili
stage=0
datadev="data/analysis1 data/analysis2 data/test_dev data/eval1 data/eval2 data/eval3"
dir=exp/chain/tdnn1b_sp
lang=data/lang_combined_chain
tree_dir=exp/chain/tree_sp
cmd=queue.pl
graph_affix=_combined

# training options
chunk_width=140,100,160
chunk_left_context=0
chunk_right_context=0

# ivector options
max_count=75 # parameter for extract_ivectors.sh
sub_speaker_frames=600
filter_ctm=true
weights_file=
silence_weight=0.00001
nj=30

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ ! -f ./conf/lang/${language}.conf ] && \
  echo "Language configuration conf/lang/${language}.conf does not exist!" && exit 1
ln -sf ./conf/lang/${language}.conf lang.conf                                   
. ./lang.conf

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le 1 ]; then
  # extract hires mfcc features from uniformly segmented data
  for datadir in $datadev; do
    utils/copy_data_dir.sh ${datadir}_segmented ${datadir}_segmented_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" ${datadir}_segmented_hires || exit 1;
    steps/compute_cmvn_stats.sh ${datadir}_segmented_hires || exit 1;
    utils/fix_data_dir.sh ${datadir}_segmented_hires || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  # extract iVectors for the test data, in this case we don't need the speed
  # perturbation (sp).
  for datadir in $datadev; do
    data=$(basename $datadir)
    steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
      --silence-weight $silence_weight \
      --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
      ${datadir}_segmented_hires $lang exp/nnet3/extractor \
      exp/nnet3/ivectors_${data}_segmented_hires
  done
fi

frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
rm $dir/.error 2>/dev/null || true

if [ $stage -le 3 ]; then
  # do the 1st pass decoding
  for datadir in $datadev; do
    (
      data=$(basename $datadir)
      nspk=$(wc -l <data/${data}_segmented_hires/spk2utt)
      decode_dir=${dir}/decode_${data}_segmented
      steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $chunk_left_context \
        --extra-right-context $chunk_right_context \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --skip-scoring true \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/nnet3/ivectors_${data}_segmented_hires \
        $tree_dir/graph${graph_affix} ${datadir}_segmented_hires ${decode_dir} || exit 1

      # resolve ctm overlaping regions, and compute wer
      local/postprocess_test.sh ${data}_segmented ${tree_dir}/graph${graph_affix} \
        ${decode_dir}
    ) || touch $dir/.error &
  done
fi
wait

if [ $stage -le 4 ]; then
  # re-segement data based on 1st-pass decoding
  segmentation_opts="--silence-proportion 0.2 --max-segment-length 15 --frame-shift 0.03"
  for datadir in $datadev; do
    data=$(basename $datadir)
    # get alignment from lattice
    nj_ali=`cat ${dir}/decode_${data}_segmented/num_jobs` || exit 1;
    $cmd JOB=1:${nj_ali} ${dir}/decode_${data}_segmented/log/generate_alignments.JOB.log \
    lattice-best-path --acoustic-scale=0.2 \
    "ark:gunzip -c ${dir}/decode_${data}_segmented/lat.JOB.gz |" \
    ark:/dev/null "ark:|gzip -c >${dir}/decode_${data}_segmented/ali.JOB.gz" || exit 1;

    cp $lang/phones.txt ${dir}/decode_${data}_segmented || exit 1;

    steps/resegment_data.sh --segmentation-opts "$segmentation_opts" ${datadir}_segmented_hires $lang \
      ${dir}/decode_${data}_segmented ${datadir}_segmented_reseg_hires_tmp exp/resegment_${data}_segmented

    utils/data/subsegment_data_dir.sh ${datadir}_segmented_hires ${datadir}_segmented_reseg_hires_tmp/segments \
      ${datadir}_segmented_reseg_hires

    rm -rf ${datadir}_segmented_reseg_hires_tmp 2>/dev/null || true

    echo "Extracting i-vectors, stage 2"
    # this does offline decoding, except we estimate the iVectors per
    # speaker, excluding silence (based on alignments from a DNN decoding), with a
    # different script.  This is just to demonstrate that script.
    # the --sub-speaker-frames is optional; if provided, it will divide each speaker
    # up into "sub-speakers" of at least that many frames... can be useful if
    # acoustic conditions drift over time within the speaker's data.
    steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
      --silence-weight $silence_weight \
      --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
      ${datadir}_segmented_reseg_hires $lang exp/nnet3/extractor \
      exp/nnet3/ivectors_${data}_segmented_reseg_hires;
  done
fi

if [ $stage -le 5 ]; then
  # 2nd-pass decoding on the resegmented data
  for datadir in $datadev; do
    (
      data=$(basename $datadir)
      nspk=$(wc -l <data/${data}_segmented_reseg_hires/spk2utt)
      decode_dir=${dir}/decode_${data}_segmented_reseg
      steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $chunk_left_context \
        --extra-right-context $chunk_right_context \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --skip-scoring true \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/nnet3/ivectors_${data}_segmented_reseg_hires \
        $tree_dir/graph${graph_affix} ${datadir}_segmented_reseg_hires ${decode_dir} || exit 1

      # resolve ctm overlaping regions, and compute wer
      local/postprocess_test.sh ${data}_segmented_reseg $tree_dir/graph${graph_affix} \
        ${decode_dir}
    ) || touch $dir/.error &
  done
fi
wait
[ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1

exit 0;
