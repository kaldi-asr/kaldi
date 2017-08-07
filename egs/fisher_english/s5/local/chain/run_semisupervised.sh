#!/bin/bash

set -e -o pipefail

stage=-2
nj=30
decode_nj=30
base_train_set=train_comb350k # for reference

unsupervised_set=train_unsup250k  # set this to your choice of unsupervised data
supervised_set=train_sup
semi_affix=350k  # affix relating train-set splitting proportion

tdnn_affix=_sup1a  # affix for the supervised chain-model directory
train_supervised_opts="--stage -10 --train-stage -10"

# combination options
decode_affix=
egs_affix=  # affix for the egs that are generated from unsupervised data and for the comined egs dir
comb_affix=_comb1a  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
unsup_frames_per_eg=  # if empty will be equal to the supervised model's config -- you will need to change minibatch_size for comb training accordingly
unsup_egs_weight=1.0
lattice_lm_scale=0.0  # lm-scale for using the weights from unsupervised lattices
lattice_prune_beam=2.0  # If supplied will prune the lattices prior to getting egs for unsupervised data
left_tolerance=2
right_tolerance=2
train_combined_opts="--num-epochs 4.5"
graph_affix=   # can be used to decode the unsup data with another lm/graph
phone_insertion_penalty=
# to tune:
# frames_per_eg for unsupervised

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

nnet3_affix=_semi${semi_affix}  # affix for nnet3 and chain dirs
decode_affix=${decode_affix}${graph_affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le -2 ]; then
  ali_dir=exp/tri4a_ali_$supervised_set
  if [ ! -f $ali_dir/ali.1.gz ]; then
    steps/align_fmllr.sh --nj 30 --cmd "queue.pl" data/$supervised_set data/lang exp/tri4a $ali_dir
  fi
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/chain/run_tdnn.sh $train_supervised_opts --remove-egs false \
                          --train-set $supervised_set \
                          --build-tree-ali-dir $ali_dir \
                          --nnet3-affix $nnet3_affix --tdnn-affix $tdnn_affix
fi

if [ $stage -le -1 ] && [ ! -f exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires/ivector_online.scp ]; then
  echo "$0: getting ivectors for the hires unsupervised data data/${unsupervised_set}_hires"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
            data/${unsupervised_set}_hires exp/nnet3${nnet3_affix}/extractor \
            exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires
fi

chaindir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp

left_context=`cat $chaindir/egs/info/left_context`
right_context=`cat $chaindir/egs/info/right_context`
left_context_initial=`cat $chaindir/egs/info/left_context_initial`
right_context_final=`cat $chaindir/egs/info/right_context_final`
[ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=`cat $chaindir/egs/info/frames_per_eg`
frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
cmvn_opts=`cat $chaindir/cmvn_opts`

if [ $stage -le 0 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $chaindir"
  graphdir=$chaindir/graph${graph_affix}
  if [ ! -f $graphdir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test${graph_affix} $chaindir $graphdir
  fi
  steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 \
            --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
            --scoring-opts "--min-lmwt 5 " \
            $chaindir/graph${graph_affix} data/${unsupervised_set}_hires $chaindir/decode_${unsupervised_set}${decode_affix}
  ln -s ../final.mdl $chaindir/decode_${unsupervised_set}${decode_affix}/final.mdl || true
fi

if [ $stage -le 1 ]; then
  echo "$0: generating egs from the unsupervised data"
  steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
             --left-tolerance $left_tolerance --right-tolerance $right_tolerance \
             --left-context $left_context --right-context $right_context \
             --left-context-initial $left_context_initial --right-context-final $right_context_final \
             --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
             --frame-subsampling-factor $frame_subsampling_factor \
             --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
             --lattice-prune-beam "$lattice_prune_beam" \
             --egs-weight $unsup_egs_weight \
             --phone-insertion-penalty "$phone_insertion_penalty" \
             --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
             data/${unsupervised_set}_hires $chaindir \
             ${chaindir}/decode_${unsupervised_set}${decode_affix} $chaindir/unsup_egs${decode_affix}${egs_affix}
fi

sup_egs_dir=$chaindir/egs
unsup_egs_dir=$chaindir/unsup_egs${decode_affix}${egs_affix}
comb_egs_dir=$chaindir/comb_egs${decode_affix}${egs_affix}
if [ $stage -le 2 ]; then
  echo "$0: combining supervised/unsupervised egs"
  n1=`cat $sup_egs_dir/info/num_archives`
  n2=`cat $unsup_egs_dir/info/num_archives`
  num_archives=$(($n2>$n1?$n2:$n1))
  num_archives=$[num_archives*3/2]
  mkdir -p $comb_egs_dir/log
  cp {$sup_egs_dir,$comb_egs_dir}/train_diagnostic.cegs
  cp {$sup_egs_dir,$comb_egs_dir}/valid_diagnostic.cegs
  nnet3-chain-copy-egs "ark:cat $sup_egs_dir/combine.cegs $unsup_egs_dir/combine.cegs |" ark:$comb_egs_dir/combine.cegs
  cp {$sup_egs_dir,$comb_egs_dir}/cmvn_opts
  cp -r $sup_egs_dir/info $comb_egs_dir
  echo $num_archives > $comb_egs_dir/info/num_archives
  cat {$sup_egs_dir,$unsup_egs_dir}/info/num_frames | awk '{s+=$1} END{print s}' > $comb_egs_dir/info/num_frames
  cat {$sup_egs_dir,$unsup_egs_dir}/info/egs_per_archive | awk '{s+=$1} END{print s}' > $comb_egs_dir/info/egs_per_archive
  out_egs_list=
  egs_list=
  for n in $(seq $num_archives); do
      [ -f $sup_egs_dir/cegs.$n.ark ] && egs_list="$egs_list $sup_egs_dir/cegs.$n.ark"
      [ -f $unsup_egs_dir/cegs.$n.ark ] && egs_list="$egs_list $unsup_egs_dir/cegs.$n.ark"
      out_egs_list="$out_egs_list ark:$comb_egs_dir/cegs.$n.ark"
  done
  srand=0
  $decode_cmd $comb_egs_dir/log/combine.log \
              nnet3-chain-copy-egs "ark:cat $egs_list|" $out_egs_list
fi

if [ $stage -le 3 ]; then
  echo "$0: training on the supervised+unsupervised subset"
  # the train-set and gmm do not matter as we are providing the egs
  local/chain/run_tdnn.sh --stage 12 --remove-egs false --train-set $supervised_set \
                          --nnet3-affix $nnet3_affix \
                          --tdnn-affix ${tdnn_affix}${decode_affix}${egs_affix}${comb_affix} \
                          --common-egs-dir $comb_egs_dir $train_combined_opts
fi
