#!/bin/bash

set -e -o pipefail


stage=0
nj=30
decode_nj=30
base_train_set=train_cleaned # the starting point train-set
base_gmm=tri3_cleaned  # the starting point of training on the supervised data (no flat start for now)
semi_affix=  # affix relating train-set splitting proportion
             # (currently supervised 25%) and the base train set (currently _cleaned), etc.
tdnn_affix=_sup1a  # affix for the supervised chain-model directory
train_supervised_opts="--stage -10 --train-stage -10"

# combination options
comb_affix=_comb1a  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
unsup_decode_lattice_beam=8.0
unsup_frames_per_eg=  # if empty will be equal to the supervised model's config
lattice_lm_scale=0.1  # lm-scale for using the weights from unsupervised lattices
left_tolerance=2
right_tolerance=2
train_combined_opts=

# to tune:
# frames_per_eg for unsupervised

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

supervised_set=${base_train_set}_sup${semi_affix}
unsupervised_set=${base_train_set}_unsup${semi_affix}
gmm=${base_gmm}_semi${semi_affix}  # the gmm to be supplied to chain/run_tdnn.sh
nnet3_affix=_cleaned_semi${semi_affix}  # affix for nnet3 and chain dirs

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le -4 ]; then
  echo "$0: preparing the supervised and unsupervised subsets of data"
  if [ -f data/$supervised_set/feats.scp ]; then
    echo "$0: supervised subset of data already exists; either delete it or use a later stage"
    exit 1;
  fi
  mkdir -p data/$supervised_set
  # get the list of supervised utts
  num_utts=`wc -l data/$base_train_set/feats.scp | cut -d' ' -f1`
  num_supervised_utts=$[num_utts/4]
  num_unsupervised_utts=$[num_utts-num_supervised_utts]
  echo "$0: spliting data/$base_train_set to supervised subset with"
  echo "    $num_supervised_utts utts and unsupervised subset with $num_unsupervised_utts utts."
  utils/shuffle_list.pl data/$base_train_set/feats.scp | cut -d' ' -f1 | \
                        head -$num_supervised_utts > data/$supervised_set/supervised_uttlist || true
  utils/shuffle_list.pl data/$base_train_set/feats.scp | cut -d' ' -f1 | \
                        tail -$num_unsupervised_utts > data/$supervised_set/unsupervised_uttlist || true
  utils/subset_data_dir.sh --utt-list data/$supervised_set/supervised_uttlist \
                           data/$base_train_set data/$supervised_set || exit 1
  utils/subset_data_dir.sh --utt-list data/$supervised_set/unsupervised_uttlist \
                           data/$base_train_set data/$unsupervised_set || exit 1
  utils/data/subset_data_dir.sh --utt-list data/$unsupervised_set/feats.scp \
                                data/${base_train_set}_sp_hires data/${unsupervised_set}_hires
fi

if [ $stage -le -3 ]; then
  # align the supervised subset with the current cleaned gmm
  if [ -f $gmm/ali.1.gz ]; then
    echo "$0: alignments in $gmm appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the supervised data data/${supervised_set}"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
                       data/${supervised_set} data/lang exp/$base_gmm exp/$gmm
fi

if [ $stage -le -2 ]; then
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/chain/run_tdnn.sh $train_supervised_opts --remove-egs false \
                          --train-set $supervised_set --gmm $gmm \
                          --nnet3-affix $nnet3_affix --tdnn-affix $tdnn_affix
fi

if [ $stage -le -1 ]; then
  echo "$0: getting ivectors for the hires unsupervised data data/${unsupervised_set}_hires"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
            data/${unsupervised_set}_hires exp/nnet3${nnet3_affix}/extractor \
            exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires
fi

chaindir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp_bi

left_context=`cat $chaindir/egs/info/left_context`
right_context=`cat $chaindir/egs/info/right_context`
left_context_initial=`cat $chaindir/egs/info/left_context_initial`
right_context_final=`cat $chaindir/egs/info/right_context_final`
[ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=`cat $chaindir/egs/info/frames_per_eg`
frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
cmvn_opts=`cat $chaindir/cmvn_opts`

if [ $stage -le 0 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $chaindir"
  steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 --lattice-beam $unsup_decode_lattice_beam \
            --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
            --scoring-opts "--min-lmwt 5 " \
            $chaindir/graph data/${unsupervised_set}_hires $chaindir/decode_${unsupervised_set}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
              data/${unsupervised_set}_hires \
              ${chaindir}/decode_${unsupervised_set} ${chaindir}/decode_${unsupervised_set}_rescore
  ln -s ../final.mdl $chaindir/decode_${unsupervised_set}_rescore/final.mdl || true
fi

if [ $stage -le 1 ]; then
  # TODO: set supervision.weight for the unsupervised data and tune it (maybe try 0.25,0.5,0.75)
  echo "$0: generating egs from the unsupervised data"
  steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
             --left-tolerance $left_tolerance --right-tolerance $right_tolerance \
             --left-context $left_context --right-context $right_context \
             --left-context-initial $left_context_initial --right-context-final $right_context_final \
             --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
             --frame-subsampling-factor $frame_subsampling_factor \
             --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
             --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
             data/${unsupervised_set}_hires $chaindir ${chaindir}/decode_${unsupervised_set}_rescore $chaindir/unsup_egs
fi

if [ $stage -le 2 ]; then
  echo "$0: combining supervised/unsupervised egs"
  num_archives=`cat $sup_chaindir/egs/info/num_archives`
  sup_egs_dir=$sup_chaindir/egs
  unsup_egs_dir=$chaindir/unsup_egs
  comb_egs_dir=$chaindir/comb_egs
  mkdir -p $comb_egs_dir/log
  cp {$sup_egs_dir,$comb_egs_dir}/train_diagnostic.cegs
  cp {$sup_egs_dir,$comb_egs_dir}/valid_diagnostic.cegs
  cp {$sup_egs_dir,$comb_egs_dir}/combine.cegs
  cp {$sup_egs_dir,$comb_egs_dir}/cmvn_opts
  cp -r $sup_egs_dir/info $comb_egs_dir
  cat {$sup_egs_dir,$unsup_egs_dir}/info/num_frames | awk '{s+=$1} END{print s}' > $comb_egs_dir/info/num_frames
  cat {$sup_egs_dir,$unsup_egs_dir}/info/egs_per_archive | awk '{s+=$1} END{print s}' > $comb_egs_dir/info/egs_per_archive
  out_egs_list=
  egs_list=
  for n in $(seq $num_archives); do
      egs_list="$egs_list $sup_egs_dir/cegs.$n.ark"
      egs_list="$egs_list $unsup_egs_dir/cegs.$n.ark"
      out_egs_list="$out_egs_list ark:$comb_egs_dir/cegs.$n.ark"
  done
  srand=0
  #    $decode_cmd --mem 8G $comb_egs_dir/log/shuffle_combine.log \
  #          nnet3-chain-shuffle-egs --srand=$srand "ark:cat $egs_list|" ark:- \| \
  #    nnet3-chain-copy-egs --random=true --srand=$srand ark:- $out_egs_list
  $decode_cmd $comb_egs_dir/log/combine.log \
        nnet3-chain-copy-egs "ark:cat $egs_list|" $out_egs_list
fi

if [ $stage -le 3 ]; then
  echo "$0: training on the supervised+unsupervised subset"
  # the train-set and gmm do not matter as we are providing the egs
  local/chain/run_tdnn.sh --stage 17 --remove-egs false --train-set $supervised_set --gmm $gmm \
        --nnet3-affix $nnet3_affix --tdnn-affix ${tdnn_affix}${comb_affix} \
        --common-egs-dir $chaindir/comb_egs $train_combined_opts
fi

#final_combined_chaindir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}${comb_affix}_sp_bi

