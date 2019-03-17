#!/bin/bash

# This script is based on swbd 7q TDNN-F recipe 
# with resnet-style skip connections, more layers,
# skinnier bottlenecks, removing the 3-way splicing and skip-layer splicing,
# and re-tuning the learning rate and l2 regularize.  The configs are
# standardized and substantially simplified.
# The advantage of this style of config is that it also works
# well on smaller datasets, and we adopt this style here also for consistency.
# This gives better results than TDNN+LSTM on AMI SDM.

# local/chain/multi_condition/tuning/run_tdnn_1a.sh --mic ihm --train-set train_cleaned --gmm tri3_cleaned &
# local/chain/multi_condition/tuning/run_tdnn_1a.sh --mic sdm1 --use-ihm-ali true --train-set train_cleaned --gmm tri3_cleaned &
# local/chain/multi_condition/tuning/run_tdnn_1a.sh --mic mdm8 --use-ihm-ali true --train-set train_cleaned --gmm tri3_cleaned &

# steps/info/chain_dir_info.pl exp/sdm1/chain_cleaned_rvb/tdnn1a_sp_rvb_bi_ihmali
# exp/sdm1/chain_cleaned_rvb/tdnn1a_sp_rvb_bi_ihmali: num-iters=193 nj=3..16 num-params=17.5M dim=40+100->3728 combine=-0.122->-0.121 (over 2) xent:train/valid[127,192,final]=(-2.03,-1.57,-1.58/-2.12,-1.71,-1.71) logprob:train/valid[127,192,final]=(-0.179,-0.121,-0.122/-0.198,-0.158,-0.157)

# local/chain/compare_wer_general.sh sdm1 chain_cleaned_rvb tdnn_lstm1b_sp_rvb_bi_ihmali tdnn1a_sp_rvb_bi_ihmali
# System                tdnn_lstm1b_sp_rvb_bi_ihmali tdnn1a_sp_rvb_bi_ihmali
# WER on dev        33.9      33.3
# WER on eval        37.4      36.7
# Final train prob      -0.133611 -0.122155
# Final valid prob      -0.161014 -0.156612
# Final train prob (xent)       -1.9774  -1.57504
# Final valid prob (xent)      -2.09991    -1.705

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
ihm_gmm=tri3_cleaned  # the gmm for the IHM system (if --use-ihm-ali true).
num_threads_ubm=32
num_data_reps=1
num_epochs=6
get_egs_stage=-5
remove_egs=false

chunk_width=160,140,110,80
dropout_schedule='0,0@0.20,0.5@0.50,0' # dropout schedule controls the dropout
                                       # proportion for each training iteration.
xent_regularize=0.1

train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1a  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.

# decode options
frames_per_chunk=160

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! $use_ihm_ali; then
  [ "$mic" != "ihm" ] && \
    echo "$0: you cannot specify --use-ihm-ali false if the microphone is not ihm." && \
    exit 1;
else
  [ "$mic" == "ihm" ] && \
    echo "$0: you must specify --use-ihm-ali false if the microphone is ihm." && \
    exit 1;
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

nnet3_affix=_cleaned
rvb_affix=_rvb


if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}_ihmdata
  original_lat_dir=exp/$mic/chain${nnet3_affix}/${ihm_gmm}_${train_set}_sp_lats_ihmdata
  lat_dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/${ihm_gmm}_${train_set}_sp${rvb_affix}_lats_ihmdata
  dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/tdnn${tdnn_affix}_sp${rvb_affix}_bi_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  gmm_dir=exp/${mic}/$gmm
  lores_train_data_dir=data/$mic/${train_set}_sp
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
  original_lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
  lat_dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/${gmm}_${train_set}_sp${rvb_affix}_lats
  dir=exp/$mic/chain${nnet3_affix}${rvb_affix}/tdnn${tdnn_affix}_sp${rvb_affix}_bi
fi


local/nnet3/multi_condition/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --nj $nj \
                                  --train-set $train_set \
                                  --num-threads-ubm $num_threads_ubm \
                                  --num-data-reps $num_data_reps \
                                  --nnet3-affix "$nnet3_affix"


# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len "" \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set


train_data_dir=data/$mic/${train_set}_sp${rvb_affix}_hires
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}${rvb_affix}/ivectors_${train_set}_sp${rvb_affix}_hires
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7


for f in $gmm_dir/final.mdl $lores_train_data_dir/feats.scp \
   $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" \
    --generate-ali-from-lats true ${lores_train_data_dir} \
    data/lang $gmm_dir $original_lat_dir
  rm $original_lat_dir/fsts.*.gz # save space

  lat_dir_ihmdata=exp/ihm/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats

  original_lat_nj=$(cat $original_lat_dir/num_jobs)
  ihm_lat_nj=$(cat $lat_dir_ihmdata/num_jobs)

  $train_cmd --max-jobs-run 10 JOB=1:$original_lat_nj $lat_dir/temp/log/copy_original_lats.JOB.log \
    lattice-copy "ark:gunzip -c $original_lat_dir/lat.JOB.gz |" ark,scp:$lat_dir/temp/lats.JOB.ark,$lat_dir/temp/lats.JOB.scp

  $train_cmd --max-jobs-run 10 JOB=1:$ihm_lat_nj $lat_dir/temp2/log/copy_ihm_lats.JOB.log \
    lattice-copy "ark:gunzip -c $lat_dir_ihmdata/lat.JOB.gz |" ark,scp:$lat_dir/temp2/lats.JOB.ark,$lat_dir/temp2/lats.JOB.scp

  for n in $(seq $original_lat_nj); do
    cat $lat_dir/temp/lats.$n.scp
  done > $lat_dir/temp/combined_lats.scp

  for i in `seq 1 $num_data_reps`; do
    for n in $(seq $ihm_lat_nj); do
      cat $lat_dir/temp2/lats.$n.scp
    done | sed -e "s/^/rev${i}_/"
  done >> $lat_dir/temp/combined_lats.scp

  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  utils/split_data.sh $train_data_dir $nj

  $train_cmd --max-jobs-run 10 JOB=1:$nj $lat_dir/copy_combined_lats.JOB.log \
    lattice-copy --include=$train_data_dir/split$nj/JOB/utt2spk \
    scp:$lat_dir/temp/combined_lats_sorted.scp \
    "ark:|gzip -c >$lat_dir/lat.JOB.gz" || exit 1;

  echo $nj > $lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $original_lat_dir/$f $lat_dir/$f
  done
fi


if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 4200 ${lores_train_data_dir} data/lang_chain $original_lat_dir $tree_dir
fi

if [ $stage -le 15 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


graph_dir=$dir/graph_${LM}
if [ $stage -le 17 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  rm $dir/.error 2>/dev/null || true

  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;

  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}${rvb_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
