#!/usr/bin/env bash

# Copyright 2012-2017  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

# Begin configuration.

config=             # config, also forwarded to 'train_scheduler.sh',

# topology, initialization,
network_type=dnn    # select type of neural network (dnn,cnn1d,cnn2d,lstm),
hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
hid_dim=1024        # number of neurons per layer,
bn_dim=             # (optional) adds bottleneck and one more hidden layer to the NN,
dbn=                # (optional) prepend layers to the initialized NN,

proto_opts=         # adds options to 'make_nnet_proto.py',
cnn_proto_opts=     # adds options to 'make_cnn_proto.py',

nnet_init=          # (optional) use this pre-initialized NN,
nnet_proto=         # (optional) use this NN prototype for initialization,

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
online_cmvn_opts=   # (optional) adds 'apply-cmvn-online' to input feature pipeline, see opts,
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
ivector=            # (optional) adds 'append-vector-to-feats', the option is rx-filename for the 2nd stream,
ivector_append_tool=append-vector-to-feats # (optional) the tool for appending ivectors,

feat_type=plain
traps_dct_basis=11    # (feat_type=traps) nr. of DCT basis, 11 is good with splice=10,
transf=               # (feat_type=transf) import this linear tranform,
splice_after_transf=5 # (feat_type=transf) splice after the linear transform,

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',

# labels,
labels=            # (optional) specify non-default training targets,
                   # (targets need to be in posterior format, see 'ali-to-post', 'feat-to-post'),
num_tgt=           # (optional) specifiy number of NN outputs, to be used with 'labels=',

# training scheduler,
learn_rate=0.008   # initial learning rate,
scheduler_opts=    # options, passed to the training scheduler,
train_tool=        # optionally change the training tool,
train_tool_opts=   # options for the training tool,
frame_weights=     # per-frame weights for gradient weighting,
utt_weights=       # per-utterance weights (scalar for --frame-weights),

# data processing, misc.
copy_feats=true     # resave the train/cv features into /tmp (disabled by default),
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
copy_feats_compress=true # compress feats while resaving
feats_std=1.0

split_feats=        # split the training data into N portions, one portion will be one 'epoch',
                    # (empty = no splitting)

seed=777            # seed value used for data-shuffling, nn-initialization, and training,
skip_cuda_check=false
skip_phoneset_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate scheduling, model selection)"
   echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --network-type (dnn,cnn1d,cnn2d,lstm)  # type of neural network"
   echo "  --nnet-proto <file>      # use this NN prototype"
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo ""
   echo "  --feat-type (plain|traps|transf) # type of input features"
   echo "  --cmvn-opts  <string>            # add 'apply-cmvn' to input feature pipeline"
   echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   echo
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo "  --copy-feats <bool>      # copy features to /tmp, lowers storage stress"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

# Using alidir for supervision (default)
if [ -z "$labels" ]; then
  silphonelist=`cat $lang/phones/silence.csl`
  for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $(cat $data/feats.scp | wc -l), $alidir \n"
printf "\t CV-set    : $data_cv $(cat $data_cv/feats.scp | wc -l) $alidir_cv \n"
echo

mkdir -p $dir/{log,nnet}

if ! $skip_phoneset_check; then
  utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt
  utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir_cv/phones.txt
  cp $lang/phones.txt $dir
fi

# skip when already trained,
if [ -e $dir/final.nnet ]; then
  echo "SKIPPING TRAINING... ($0)"
  echo "nnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))"
  exit 0
fi

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  # training targets in posterior format,
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  # training targets for analyze-counts,
  labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"

  # get pdf-counts, used later for decoding/aligning,
  num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
  analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    ${utt_weights:+ "--utt-weights=$utt_weights"} \
    "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log
  # copy the old transition model, will be needed by decoder,
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl
  # copy the tree
  cp $alidir/tree $dir/tree

  # make phone counts for analysis,
  [ -e $lang/phones.txt ] && analyze-counts --verbose=1 --symbol-table=$lang/phones.txt --counts-dim=$num_pdf \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    ${utt_weights:+ "--utt-weights=$utt_weights"} \
    "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
if [ "$copy_feats" == "true" ]; then
  echo "# re-saving features to local disk,"
  tmpdir=$(mktemp -d $copy_feats_tmproot)
  copy-feats --compress=$copy_feats_compress scp:$data/feats.scp ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp
  copy-feats --compress=$copy_feats_compress scp:$data_cv/feats.scp ark,scp:$tmpdir/cv.ark,$dir/cv.scp
  trap "echo '# Removing features tmpdir $tmpdir @ $(hostname)'; ls $tmpdir; rm -r $tmpdir" EXIT
else
  # or copy the list,
  cp $data/feats.scp $dir/train_sorted.scp
  cp $data_cv/feats.scp $dir/cv.scp
fi
# shuffle the list,
utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

# create a 10k utt subset for global cmvn estimates,
head -n 10000 $dir/train.scp > $dir/train.scp.10k

# split the list,
if [ -n "$split_feats" ]; then
  scps= # 1..split_feats,
  for (( ii=1; ii<=$split_feats; ii++ )); do scps="$scps $dir/train.${ii}.scp"; done
  utils/split_scp.pl $dir/train.scp $scps
fi

# for debugging, add lists with non-local features,
utils/shuffle_list.pl --srand ${seed:-777} <$data/feats.scp >$dir/train.scp_non_local
cp $data_cv/feats.scp $dir/cv.scp_non_local

###### OPTIONALLY IMPORT FEATURE SETTINGS (from pre-training) ######
ivector_dim= # no ivectors,
if [ -n "$feature_transform" ]; then
  D=$(dirname $feature_transform)
  echo "# importing feature settings from dir '$D'"
  [ -e $D/online_cmvn_opts ] && online_cmvn_opts=$(cat $D/online_cmvn_opts)
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  [ -e $D/ivector_dim ] && ivector_dim=$(cat $D/ivector_dim)
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  echo "# cmvn_opts='$cmvn_opts' delta_opts='$delta_opts' ivector_dim='$ivector_dim'"
fi

###### PREPARE FEATURE PIPELINE ######
# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"

# optionally add per-speaker CMVN,
[ -n "$online_cmvn_opts" -a -n "$cmvn_opts" ] && echo "Error: use \$online_cmvn_opts or \$cmvn_opts, not both!" && exit 1
if [ -n "$online_cmvn_opts" ]; then
  echo "# + 'apply-cmvn-online' with '$online_cmvn_opts' is used,"
  global_cmvn_stats=$dir/global_cmvn_stats.mat
  matrix-sum --binary=false scp:$data/cmvn.scp $global_cmvn_stats
  feats_tr="$feats_tr apply-cmvn-online $online_cmvn_opts $global_cmvn_stats ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn-online $online_cmvn_opts $global_cmvn_stats ark:- ark:- |"
elif [ -n "$cmvn_opts" ]; then
  echo "# + 'apply-cmvn' with '$cmvn_opts' using statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
else
  echo "# 'apply-cmvn' is not used,"
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
fi

# keep track of the config,
[ -n "$online_cmvn_opts" ] && echo "$online_cmvn_opts" >$dir/online_cmvn_opts
[ -n "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts
[ -n "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# temoprary pipeline with first 10k,
feats_tr_10k="${feats_tr/train.scp/train.scp.10k}"

# get feature dim,
feat_dim=$(feat-to-dim "$feats_tr_10k" -)
echo "# feature dim : $feat_dim (input of 'feature_transform')"

# Now we start building 'feature_transform' which goes right in front of a NN.
# The forwarding is computed on a GPU before the frame shuffling is applied.
#
# Same GPU is used both for 'feature_transform' and the NN training.
# So it has to be done by a single process (we are using exclusive mode).
# This also reduces the CPU-GPU uploads/downloads to minimum.

if [ -n "$feature_transform" ]; then
  echo "# importing 'feature_transform' from '$feature_transform'"
  tmp=$dir/imported_$(basename $feature_transform)
  cp $feature_transform $tmp; feature_transform=$tmp
else
  # Make default proto with splice,
  if [ -n "$feature_transform_proto" ]; then
    echo "# importing custom 'feature_transform_proto' from '$feature_transform_proto'"
  else
    echo "# + default 'feature_transform_proto' with splice +/-$splice frames,"
    feature_transform_proto=$dir/splice${splice}.proto
    echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>" >$feature_transform_proto
  fi

  # Initialize 'feature-transform' from a prototype,
  feature_transform=$dir/tr_$(basename $feature_transform_proto .proto).nnet
  nnet-initialize --binary=false $feature_transform_proto $feature_transform

  # Choose further processing of spliced features
  echo "# feature type : $feat_type"
  case $feat_type in
    plain)
    ;;
    traps)
      #generate hamming+dct transform
      feature_transform_old=$feature_transform
      feature_transform=${feature_transform%.nnet}_hamm_dct${traps_dct_basis}.nnet
      echo "# + Hamming DCT transform (t$((splice*2+1)),dct${traps_dct_basis}) into '$feature_transform'"
      #prepare matrices with time-transposed hamming and dct
      utils/nnet/gen_hamm_mat.py --fea-dim=$feat_dim --splice=$splice > $dir/hamm.mat
      utils/nnet/gen_dct_mat.py --fea-dim=$feat_dim --splice=$splice --dct-basis=$traps_dct_basis > $dir/dct.mat
      #put everything together
      compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat - | \
        transf-to-nnet - - | \
        nnet-concat --binary=false $feature_transform_old - $feature_transform
    ;;
    transf)
      feature_transform_old=$feature_transform
      feature_transform=${feature_transform%.nnet}_transf_splice${splice_after_transf}.nnet
      [ -z $transf ] && transf=$alidir/final.mat
      [ ! -f $transf ] && echo "Missing transf $transf" && exit 1
      feat_dim=$(feat-to-dim "$feats_tr_10k nnet-forward 'nnet-concat $feature_transform_old \"transf-to-nnet $transf - |\" - |' ark:- ark:- |" -)
      nnet-concat --binary=false $feature_transform_old \
        "transf-to-nnet $transf - |" \
        "utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice_after_transf |" \
        $feature_transform
    ;;
    *)
      echo "Unknown feature type $feat_type"
      exit 1;
    ;;
  esac

  # keep track of feat_type,
  echo $feat_type > $dir/feat_type

  # Renormalize the MLP input to zero mean and unit variance,
  feature_transform_old=$feature_transform
  feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
  echo "# compute normalization stats from 10k sentences"
  nnet-forward --print-args=true --use-gpu=yes $feature_transform_old \
    "$feats_tr_10k" ark:- |\
    compute-cmvn-stats ark:- $dir/cmvn-g.stats
  echo "# + normalization of NN-input at '$feature_transform'"
  nnet-concat --binary=false $feature_transform_old \
    "cmvn-to-nnet --std-dev=$feats_std $dir/cmvn-g.stats -|" $feature_transform
fi

if [ ! -z $ivector ]; then
  echo
  echo "# ADDING IVECTOR FEATURES"
  # The iVectors are concatenated 'as they are' directly to the input of the neural network,
  # To do this, we paste the features, and use <ParallelComponent> where the 1st component
  # contains the transform and 2nd network contains <Copy> component.

  echo "# getting dims,"
  dim_raw=$(feat-to-dim "$feats_tr_10k" -)
  dim_raw_and_ivec=$(feat-to-dim "$feats_tr_10k $ivector_append_tool ark:- '$ivector' ark:- |" -)
  dim_ivec=$((dim_raw_and_ivec - dim_raw))
  echo "# dims, feats-raw $dim_raw, ivectors $dim_ivec,"

  # Should we do something with 'feature_transform'?
  if [ ! -z $ivector_dim ]; then
    # No, the 'ivector_dim' comes from dir with 'feature_transform' with iVec forwarding,
    echo "# assuming we got '$feature_transform' with ivector forwarding,"
    [ $ivector_dim != $dim_ivec ] && \
    echo -n "Error, i-vector dimensionality mismatch!" && \
    echo " (expected $ivector_dim, got $dim_ivec in $ivector)" && exit 1
  else
    # Yes, adjust the transform to do ``iVec forwarding'',
    feature_transform_old=$feature_transform
    feature_transform=${feature_transform%.nnet}_ivec_copy.nnet
    echo "# setting up ivector forwarding into '$feature_transform',"
    dim_transformed=$(feat-to-dim "$feats_tr_10k nnet-forward $feature_transform_old ark:- ark:- |" -)
    nnet-initialize --print-args=false <(echo "<Copy> <InputDim> $dim_ivec <OutputDim> $dim_ivec <BuildVector> 1:$dim_ivec </BuildVector>") $dir/tr_ivec_copy.nnet
    nnet-initialize --print-args=false <(echo "<ParallelComponent> <InputDim> $((dim_raw+dim_ivec)) <OutputDim> $((dim_transformed+dim_ivec)) \
                                               <NestedNnetFilename> $feature_transform_old $dir/tr_ivec_copy.nnet </NestedNnetFilename>") $feature_transform
  fi
  echo $dim_ivec >$dir/ivector_dim # mark down the iVec dim!
  echo $ivector_append_tool >$dir/ivector_append_tool

  # pasting the iVecs to the features,
  echo "# + ivector input '$ivector'"
  feats_tr="$feats_tr $ivector_append_tool ark:- '$ivector' ark:- |"
  feats_cv="$feats_cv $ivector_append_tool ark:- '$ivector' ark:- |"
fi

###### Show the final 'feature_transform' in the log,
echo
echo "### Showing the final 'feature_transform':"
nnet-info $feature_transform
echo "###"

###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
[ -f $dir/final.feature_transform ] && unlink $dir/final.feature_transform
(cd $dir; ln -s $(basename $feature_transform) final.feature_transform )
feature_transform=$dir/final.feature_transform


###### INITIALIZE THE NNET ######
echo
echo "# NN-INITIALIZATION"
if [ ! -z $nnet_init ]; then
  echo "# using pre-initialized network '$nnet_init'"
elif [ ! -z $nnet_proto ]; then
  echo "# initializing NN from prototype '$nnet_proto'";
  nnet_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  nnet-initialize --seed=$seed $nnet_proto $nnet_init
else
  echo "# getting input/output dims :"
  # input-dim,
  get_dim_from=$feature_transform
  [ ! -z "$dbn" ] && get_dim_from="nnet-concat $feature_transform '$dbn' -|"
  num_fea=$(feat-to-dim "$feats_tr_10k nnet-forward \"$get_dim_from\" ark:- ark:- |" -)

  # output-dim,
  [ -z $num_tgt ] && \
    num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  # make network prototype,
  nnet_proto=$dir/nnet.proto
  echo "# genrating network prototype $nnet_proto"
  case "$network_type" in
    dnn)
      utils/nnet/make_nnet_proto.py $proto_opts \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        $num_fea $num_tgt $hid_layers $hid_dim >$nnet_proto
      ;;
    cnn1d)
      delta_order=$([ -z $delta_opts ] && echo "0" || { echo $delta_opts | tr ' ' '\n' | grep "delta[-_]order" | sed 's:^.*=::'; })
      echo "Debug : $delta_opts, delta_order $delta_order"
      utils/nnet/make_cnn_proto.py $cnn_proto_opts \
        --splice=$splice --delta-order=$delta_order --dir=$dir \
        $num_fea >$nnet_proto
      cnn_fea=$(cat $nnet_proto | grep -v '^$' | tail -n1 | awk '{ print $5; }')
      utils/nnet/make_nnet_proto.py $proto_opts \
        --no-smaller-input-weights \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        "$cnn_fea" $num_tgt $hid_layers $hid_dim >>$nnet_proto
      ;;
    lstm)
      utils/nnet/make_lstm_proto.py $proto_opts \
        $num_fea $num_tgt >$nnet_proto
      ;;
    blstm)
      utils/nnet/make_blstm_proto.py $proto_opts \
        $num_fea $num_tgt >$nnet_proto
      ;;
    *) echo "Unknown : --network-type $network_type" && exit 1;
  esac

  # initialize,
  nnet_init=$dir/nnet.init
  echo "# initializing the NN '$nnet_proto' -> '$nnet_init'"
  nnet-initialize --seed=$seed $nnet_proto $nnet_init

  # optionally prepend dbn to the initialization,
  if [ ! -z "$dbn" ]; then
    nnet_init_old=$nnet_init; nnet_init=$dir/nnet_dbn_dnn.init
    nnet-concat "$dbn" $nnet_init_old $nnet_init
  fi
fi


###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet/train_scheduler.sh \
  ${scheduler_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${train_tool_opts:+ --train-tool-opts "$train_tool_opts"} \
  ${feature_transform:+ --feature-transform $feature_transform} \
  ${split_feats:+ --split-feats $split_feats} \
  --learn-rate $learn_rate \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${utt_weights:+ --utt-weights "$utt_weights"} \
  ${config:+ --config $config} \
  $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir

echo "$0: Successfuly finished. '$dir'"

sleep 3
exit 0
