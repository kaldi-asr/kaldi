#!/usr/bin/env bash
# Copyright   2019   Ashish Arora, Vimal Manohar, Desh Raj
# Apache 2.0.
# This script is similar to the decode_diarized_css.sh script, except that 
# it decodes streams before diarization with a speaker-indpendent
# acoustic model, so no i-vector estimation is required. As such, it
# performs a single-pass decoding instead of the 2-pass decoding in the
# other decoding script.

stage=0
nj=8
cmd=queue.pl
lm_suffix=

# decode opts
lattice_beam=8
extra_left_context=0 # change for (B)LSTM
extra_right_context=0 # change for BLSTM
frames_per_chunk=150 # change for (B)LSTM
acwt=0.1 # important to change this when using chain models
post_decode_acwt=1.0 # important to change this when using chain models
extra_left_context_initial=0
extra_right_context_final=0

graph_affix=

score_opts="--min-lmwt 6 --max-lmwt 13"

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "Usage: $0 <in-data-dir> <lang-dir> <model-dir> <out-dir>"
  echo "e.g.: $0 data/dev data/lang_chain exp/chain/tdnn_1a \
                 data/dev_segmented"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_in=$1
lang_dir=$2
asr_model_dir=$3
out_dir=$4

for f in $data_in/wav.scp $data_in/text.bak \
         $lang_dir/L.fst $asr_model_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  echo "$0 copying data files in output directory"
  mkdir -p ${out_dir}
  cp ${data_in}/{wav.scp,utt2spk.bak,segments,utt2spk,spk2utt} ${out_dir}/
  utils/data/get_reco2dur.sh ${out_dir}
fi

if [ $stage -le 1 ]; then
  # Now we extract features
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd queue.pl ${out_dir}
  steps/compute_cmvn_stats.sh ${out_dir}
  cp $data_in/text.bak ${out_dir}/text
fi

if [ $stage -le 2 ]; then
  utils/mkgraph.sh \
      --self-loop-scale 1.0 $lang_dir \
      $asr_model_dir $asr_model_dir/graph${lm_suffix}
fi

data_set=$(basename $out_dir)
decode_dir=$asr_model_dir/decode_${data_set}
# generate the lattices
if [ $stage -le 3 ]; then
  echo "Generating lattices, stage 1"
  steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
    --acwt $acwt --post-decode-acwt $post_decode_acwt \
    --extra-left-context $extra_left_context  \
    --extra-right-context $extra_right_context  \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk "$frames_per_chunk" \
    --skip-scoring true ${iter:+--iter $iter} \
    $asr_model_dir/graph${lm_suffix} ${out_dir} ${decode_dir};
fi

exit 0
