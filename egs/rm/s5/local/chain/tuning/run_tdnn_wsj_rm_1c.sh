#!/usr/bin/env bash
# _1c is as _1b but it uses source chain-trained DNN model instead of GMM model
# to generate alignments for RM using WSJ model.

# _1b is as _1a, but different as follows
# 1) It uses wsj phone set phones.txt and new lexicon generated using word pronunciation
#    in wsj lexicon.txt. rm words, that are not presented in wsj, are added as oov
#    in new lexicon.txt.
# 2) It uses wsj tree-dir and generates new alignments and lattices for rm using
#    wsj gmm model.
# 3) It also trains phone LM using weighted combination of alignemts from wsj
#    and rm, which is used in chain denominator graph.
#    Since we use phone.txt from source dataset, this can be helpful in cases
#    where there is a few training data in the target domain and some 4-gram phone
#    sequences have no count in the target domain.
# 4) It transfers all layers in already-trained model and
#    re-train the last layer using target dataset, instead of replacing it
#    with new randomely initialized output layer.

# This script uses weight transfer as Transfer learning method
# and use already trained model on wsj and fine-tune the whole network using
# rm data while training the last layer with higher learning-rate.
# The chain config is as run_tdnn_5n.sh and the result is:
# System tdnn_5n tdnn_wsj_rm_1a tdnn_wsj_rm_1b tdnn_wsj_rm_1c
# WER      2.71     1.68            3.56          3.54

set -e

# configs for 'chain'
stage=0
train_stage=-4
get_egs_stage=-10
dir=exp/chain/tdnn_wsj_rm_1c

# configs for transfer learning

common_egs_dir=
primary_lr_factor=0.25 # learning-rate factor for all except last layer in transferred source model
nnet_affix=_online_wsj

phone_lm_scales="1,10" # comma-separated list of positive integer multiplicities
                       # to apply to the different source data directories (used
                       # to give the RM data a higher weight).

# model and dirs for source model used for transfer learning
src_mdl=../../wsj/s5/exp/chain/tdnn1d_sp/final.mdl # input chain model
                                                    # trained on source dataset (wsj) and
                                                    # this model is transfered to the target domain.

src_mfcc_config=../../wsj/s5/conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features used for ivector training
                                                  # in source domain.
src_ivec_extractor_dir=  # source ivector extractor dir used to extract ivector for
                         # source data and the ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in source model training.

src_lang=../../wsj/s5/data/lang # source lang directory used to train source model.
                                # new lang dir for transfer learning experiment is prepared
                                # using source phone set phone.txt and lexicon.txt in src lang dir and
                                # word.txt target lang dir.
src_dict=../../wsj/s5/data/local/dict_nosp  # dictionary for source dataset containing lexicon.txt,
                                            # nonsilence_phones.txt,...
                                            # lexicon.txt used to generate lexicon.txt for
                                            # src-to-tgt transfer.

src_tree_dir=../../wsj/s5/exp/chain/tree_a_sp # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

# dirs for src-to-tgt transfer experiment
lang_dir=data/lang_chain_5n   # lang dir for target data.
lang_src_tgt=data/lang_wsj_rm # This dir is prepared using phones.txt and lexicon from
                              # source(WSJ) and and wordlist and G.fst from target(RM)
lat_dir=exp/chain_lats_wsj

required_files="$src_mfcc_config $src_mdl $src_lang/phones.txt $src_dict/lexicon.txt $src_tree_dir/tree"

use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is "
    echo "specified but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi


for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f" && exit 1;
  fi
done

if [ $stage -le -1 ]; then
  echo "$0: Prepare lang for RM-WSJ using WSJ phone set and lexicon and RM word list."
  if ! cmp -s <(grep -v "^#" $src_lang/phones.txt) <(grep -v "^#" $lang_dir/phones.txt); then
    local/prepare_wsj_rm_lang.sh  $src_dict $src_lang $lang_src_tgt || exit 1;
  else
    rm -rf $lang_src_tgt 2>/dev/null || true
    cp -r $lang_dir $lang_src_tgt
  fi
fi

local/online/run_nnet2_common.sh  --stage $stage \
                                  --ivector-dim $ivector_dim \
                                  --nnet-affix "$nnet_affix" \
                                  --mfcc-config $src_mfcc_config \
                                  --extractor $src_ivec_extractor_dir || exit 1;
src_mdl_dir=`dirname $src_mdl`
ivec_opt=""
if $use_ivector;then ivec_opt="--online-ivector-dir exp/nnet2${nnet_affix}/ivectors" ; fi

if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/nnet3/align_lats.sh --nj 100 --cmd "$train_cmd" $ivec_opt \
    --generate-ali-from-lats true \
    --acoustic-scale 1.0 --extra-left-context-initial 0 --extra-right-context-final 0 \
    --frames-per-chunk 150 \
    --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
    data/train_hires $lang_src_tgt $src_mdl_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 5 ]; then
  # Set the learning-rate-factor for all transferred layers but the last output
  # layer to primary_lr_factor.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
      $src_mdl $dir/input.raw || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $src_tree_dir $lat_dir $dir || exit 1;
fi

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  # exclude phone_LM and den.fst generation training stage
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi

  ivector_dir=
  if $use_ivector; then ivector_dir="exp/nnet2${nnet_affix}/ivectors" ; fi

  # we use chain model from source to generate lats for target and the
  # tolerance used in chain egs generation using this lats should be 1 or 2 which is
  # (source_egs_tolerance/frame_subsampling_factor)
  # source_egs_tolerance = 5
  chain_opts=(--chain.alignment-subsampling-factor=1 --chain.left-tolerance=1 --chain.right-tolerance=1)
  steps/nnet3/chain/train.py --stage $train_stage ${chain_opts[@]} \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir "$ivector_dir" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.frames-per-iter 1000000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/train_hires \
    --tree-dir $src_tree_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi

if [ $stage -le 8 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  tes_ivec_opt=""
  if $use_ivector;then test_ivec_opt="--online-ivector-dir exp/nnet2${nnet_affix}/ivectors_test" ; fi

  utils/mkgraph.sh --self-loop-scale 1.0 $lang_src_tgt $dir $dir/graph
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --scoring-opts "--min-lmwt 1" \
    --nj 20 --cmd "$decode_cmd" $test_ivec_opt \
    $dir/graph data/test_hires $dir/decode || exit 1;
fi
wait;
exit 0;
