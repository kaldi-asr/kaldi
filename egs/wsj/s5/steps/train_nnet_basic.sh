#!/bin/bash

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

proto_opts="--activation-final=<Sigmoid> --no-softmax"         # adds options to 'make_nnet_proto.py',
cnn_proto_opts=     # adds options to 'make_cnn_proto.py',

nnet_init=          # (optional) use this pre-initialized NN,
nnet_proto=         # (optional) use this NN prototype for initialization,

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
ivector=            # (optional) adds 'append-vector-to-feats', the option is rx-filename for the 2nd stream,
ivector_append_tool=append-vector-to-feats # (optional) the tool for appending ivectors,

## custom features for idlak support
insplice=
indelta_opts=
minmax_opts=
incmvn_opts="--norm-means=true --norm-vars=true"
##

feat_type=plain
traps_dct_basis=11    # (feat_type=traps) nr. of DCT basis, 11 is good with splice=10,
transf=               # (feat_type=transf) import this linear tranform,
splice_after_transf=5 # (feat_type=transf) splice after the linear transform,


input_feature_transform_proto= # (optional) use this prototype for 'input_feature_transform',
input_feature_transform= # (optional) set the feature transform in front of the trained MLP
speak_transform=         # (optional) fmllr transformations

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',
pytel_transform=    # (BUT) use external python transform,

num_tgt=           # (optional) specifiy number of NN outputs, to be used with 'labels=',

# training scheduler,
learn_rate=0.008   # initial learning rate,
scheduler_opts=    # options, passed to the training scheduler,
train_tool=nnet-train-frmshuff-tgtmat  # default idlak training tool
                   # optionally change the training tool,
train_tool_opts="--objective-function=mse"   # options for the training tool,
frame_weights=     # per-frame weights for gradient weighting,
utt_weights=       # per-utterance weights (scalar for --frame-weights),

# data processing, misc.
copy_feats=        # resave the train/cv features into /tmp (disabled by default),
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
copy_feats_compress=true # compress feats while resaving
feats_std=1.0

split_feats=        # split the training data into N portions, one portion will be one 'epoch',
                    # (empty = no splitting)

seed=777            # seed value used for data-shuffling, nn-initialization, and training,
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 5 ]; then
   echo "Usage: $0 <indata-train> <indata-dev> <outdata-train> <outdata-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/out data/out_cv exp/mono_nnet"
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
odata=$3
odata_cv=$4
dir=$5

for f in $data/feats.scp $data_cv/feats.scp $odata/feats.scp $odata_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $(cat $data/feats.scp | wc -l), $odata \n"
printf "\t CV-set    : $data_cv $(cat $data_cv/feats.scp | wc -l) $odata_cv \n"
echo

mkdir -p $dir/{log,nnet}

# skip when already trained,
if [ -e $dir/final.nnet ]; then
  echo "SKIPPING TRAINING... ($0)"
  echo "nnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))"
  exit 0
fi

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
if [ "$copy_feats" == "true" ]; then
  echo "# re-saving features to local disk,"
  tmpdir=$(mktemp -d $copy_feats_tmproot)
  copy-feats --compress=$copy_feats_compress scp:$data/feats.scp ark,scp:$tmpdir/intrain.ark,$dir/intrain_sorted.scp
  copy-feats --compress=$copy_feats_compress scp:$data_cv/feats.scp ark,scp:$tmpdir/incv.ark,$dir/incv.scp
  copy-feats --compress=$copy_feats_compress scp:$odata/feats.scp ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp
  copy-feats --compress=$copy_feats_compress scp:$odata_cv/feats.scp ark,scp:$tmpdir/cv.ark,$dir/cv.scp
  trap "echo '# Removing features tmpdir $tmpdir @ $(hostname)'; ls $tmpdir; rm -r $tmpdir" EXIT
else
  # or copy the list,
  cp $data/feats.scp $dir/intrain_sorted.scp
  cp $data_cv/feats.scp $dir/incv.scp
  cp $odata/feats.scp $dir/train_sorted.scp
  cp $odata_cv/feats.scp $dir/cv.scp
fi
# shuffle the list,
utils/shuffle_list.pl --srand ${seed:-777} <$dir/intrain_sorted.scp >$dir/intrain.scp
utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

# create a 10k utt subset for global cmvn estimates,
head -n 10000 $dir/train.scp > $dir/train.scp.10k
head -n 10000 $dir/intrain.scp > $dir/intrain.scp.10k

# split the list,
if [ -n "$split_feats" ]; then
  iscps=
  scps= # 1..split_feats,
  for (( ii=1; ii<=$split_feats; ii++ )); do
      scps="$scps $dir/train.${ii}.scp"; iscps="$iscps $dir/intrain.${ii}.scp" 
  done
  utils/split_scp.pl $dir/train.scp $scps
  utils/split_scp.pl $dir/intrain.scp $iscps
fi

# for debugging, add lists with non-local features,
utils/shuffle_list.pl --srand ${seed:-777} <$data/feats.scp >$dir/train.scp_non_local
cp $data_cv/feats.scp $dir/cv.scp_non_local

###### OPTIONALLY IMPORT FEATURE SETTINGS (from pre-training) ######
ivector_dim= # no ivectors,
if [ ! -z $feature_transform ]; then
  D=$(dirname $feature_transform)
  echo "# importing feature settings from dir '$D'"
  [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  [ -e $D/ivector_dim ] && ivector_dim=$(cat $D/ivector_dim)
  [ -e $D/ivector_append_tool ] && ivector_append_tool=$(cat $D/ivector_append_tool)
  echo "# cmvn_opts='$cmvn_opts' delta_opts='$delta_opts' ivector_dim='$ivector_dim'"
fi

###### PREPARE FEATURE PIPELINE ######
# input features,
# HACKY: we merge the optional input feature transform in the pipeline 
if [ ! -z $input_feature_transform ]; then
  cp $input_feature_transform $dir/input_final.feature_transform
  # Removed $gpu_option_off
  infeats_tr="ark:nnet-forward $input_feature_transform scp:$dir/intrain.scp ark:- |"
  infeats_cv="ark:nnet-forward $input_feature_transform scp:$dir/incv.scp ark:- |"
else
  infeats_tr="ark:copy-feats scp:$dir/intrain.scp ark:- |"
  infeats_cv="ark:copy-feats scp:$dir/incv.scp ark:- |"

  # optionally add deltas to input,
  if [ ! -z "$indelta_opts" ]; then
    infeats_tr="$infeats_tr add-deltas $indelta_opts ark:- ark:- |"
    infeats_cv="$infeats_cv add-deltas $indelta_opts ark:- ark:- |"
    echo "# + 'add-deltas' with '$indelta_opts'"
  fi

  # Optionally apply cmvn
  #renormalize the MLP output to zero mean and unit variance
  if [ ! -z "$incmvn_opts" ]; then
      echo "Computing global cmvn on input"
      compute-cmvn-stats --binary=false "$infeats_tr" $dir/incmvn_glob.ark
      infeats_tr="$infeats_tr apply-cmvn $incmvn_opts $dir/incmvn_glob.ark ark:- ark:- |"
	  infeats_cv="$infeats_cv apply-cmvn $incmvn_opts $dir/incmvn_glob.ark ark:- ark:- |"
  else
      echo "No CMVN used on MLP front-end"
  fi
  if [ ! -z $input_feature_transform_proto ]; then
    echo "# importing custom 'feature_transform_proto' from '$input_feature_transform_proto'"
  else
    raw_dim=$(feat-to-dim "$infeats_tr" -);
    insplice=${insplice:-0}
    echo "# + default 'input_feature_transform_proto' with splice +/-$insplice frames,"
    input_feature_transform_proto=$dir/insplice${insplice}.proto
    echo "<Splice> <InputDim> $raw_dim <OutputDim> $(((2*insplice+1)*raw_dim)) <BuildVector> -$insplice:$insplice </BuildVector>" >$input_feature_transform_proto
  fi
  # Initialize 'input_feature-transform' from a prototype,
  input_feature_transform=$dir/tr_$(basename $input_feature_transform_proto .proto).nnet
  nnet-initialize --binary=false $input_feature_transform_proto $input_feature_transform
  infeats_tr="$infeats_tr nnet-forward $input_feature_transform ark:- ark:- |"
  infeats_cv="$infeats_cv nnet-forward $input_feature_transform ark:- ark:- |"
  cp $input_feature_transform $dir/input_final.feature_transform
fi

# output features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"


# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "# + 'apply-cmvn' with '$cmvn_opts' using statistics : $odata/cmvn.scp, $odata_cv/cmvn.scp"
  [ ! -r $odata/cmvn.scp ] && echo "Missing $odata/cmvn.scp" && exit 1;
  [ ! -r $odata_cv/cmvn.scp ] && echo "Missing $odata_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$odata/utt2spk scp:$odata/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$odata_cv/utt2spk scp:$odata_cv/cmvn.scp ark:- ark:- |"
else
  echo "# 'apply-cmvn' is not used,"
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
fi

# optionally add fmllr transformations,
if [ ! -z $speak_transform ] && [ -f $speak_transform ]; then
   feats_tr="$feats_tr transform-feats --utt2spk=ark:$data/utt2spk ark:$speak_transform ark:- ark:- |"
   feats_cv="$feats_cv transform-feats --utt2spk=ark:$data_cv/utt2spk ark:$speak_transform ark:- ark:- |"
   echo "# added fmllr from $speak_transform"
fi



# keep track of the config,
[ ! -z "$indelta_opts" ] && echo "$indelta_opts" > $dir/indelta_opts
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts
[ ! -z "$incmvn_opts" ] && echo "$incmvn_opts" >$dir/incmvn_opts
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# optionally append python feature transform,
if [ ! -z "$pytel_transform" ]; then
  cp $pytel_transform $dir/pytel_transform.py
  { echo; echo "### Comes from here: '$pytel_transform' ###"; } >> $dir/pytel_transform.py
  pytel_transform=$dir/pytel_transform.py
  feats_tr="$feats_tr /bin/env python $pytel_transform |"
  feats_cv="$feats_cv /bin/env python $pytel_transform |"
  echo "# + 'pytel-transform' from '$pytel_transform'"
fi

# temporary pipeline with first 10k,
feats_tr_10k="${feats_tr/train.scp/train.scp.10k}"
infeats_tr_10k="${infeats_tr/intrain.scp/intrain.scp.10k}"

# get feature dim,
feat_dim=$(feat-to-dim "$feats_tr_10k" -)
echo "# feature dim : $feat_dim (input of 'feature_transform')"

# Now we start building 'feature_transform' which goes right in front of a NN.
# The forwarding is computed on a GPU before the frame shuffling is applied.
#
# Same GPU is used both for 'feature_transform' and the NN training.
# So it has to be done by a single process (we are using exclusive mode).
# This also reduces the CPU-GPU uploads/downloads to minimum.

if [ ! -z "$feature_transform" ]; then
  echo "# importing 'feature_transform' from '$feature_transform'"
  tmp=$dir/imported_$(basename $feature_transform)
  cp $feature_transform $tmp; feature_transform=$tmp
else
  # Make default proto with splice,
  if [ ! -z $feature_transform_proto ]; then
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

  # Renormalize the MLP output with either minmax (i.e. all values shrank to range [0:1]),
  # or zero mean and unit variance
  if [ ! -z $minmax_opts ]; then
    feature_transform_old=$feature_transform
    feature_transform=${feature_transform%.nnet}_minmax.nnet
    echo "Renormalizing MLP output features using minmax into $feature_transform"
    nnet-forward $feature_transform_old "$feats_tr" ark:- |\
      compute-minmax-stats ark:- - 2>$dir/log/minmax_calculation.log | minmax-to-nnet - - |\
      nnet-concat --binary=false $feature_transform_old - $feature_transform
  else
    # Renormalize the MLP input to zero mean and unit variance,
    feature_transform_old=$feature_transform
    feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
    echo "# compute normalization stats from 10k sentences"
    nnet-forward --print-args=true --use-gpu=yes $feature_transform_old \
        "$feats_tr_10k" ark:- |\
    compute-cmvn-stats --binary=false ark:- $dir/cmvn-g.stats
    echo "# + normalization of NN-output at '$feature_transform'"
    nnet-concat --binary=false $feature_transform_old \
        "cmvn-to-nnet --std-dev=$feats_std $dir/cmvn-g.stats -|" $feature_transform
  fi
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

  # pasting the iVecs to the feaures,
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
  num_in=$(feat-to-dim "$infeats_tr_10k" -);

  # output-dim,
  num_out=$(feat-to-dim "$feats_tr nnet-forward $feature_transform ark:- ark:- |" - )

  # make network prototype,
  nnet_proto=$dir/nnet.proto
  echo "# genrating network prototype $nnet_proto"
  case "$network_type" in
    dnn)
      train_tool_opts+=" --randomize=true"
      utils/nnet/make_nnet_proto.py $proto_opts \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        $num_in $num_out $hid_layers $hid_dim >$nnet_proto
      ;;
    cnn1d)
      delta_order=$([ -z $delta_opts ] && echo "0" || { echo $delta_opts | tr ' ' '\n' | grep "delta[-_]order" | sed 's:^.*=::'; })
      echo "Debug : $delta_opts, delta_order $delta_order"
      utils/nnet/make_cnn_proto.py $cnn_proto_opts \
        --splice=$splice --delta-order=$delta_order --dir=$dir \
        $num_in >$nnet_proto
      cnn_fea=$(cat $nnet_proto | grep -v '^$' | tail -n1 | awk '{ print $5; }')
      utils/nnet/make_nnet_proto.py $proto_opts \
        --no-smaller-input-weights \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        "$cnn_fea" $num_out $hid_layers $hid_dim >>$nnet_proto
      ;;
    cnn2d)
      delta_order=$([ -z $delta_opts ] && echo "0" || { echo $delta_opts | tr ' ' '\n' | grep "delta[-_]order" | sed 's:^.*=::'; })
      echo "Debug : $delta_opts, delta_order $delta_order"
      utils/nnet/make_cnn2d_proto.py $cnn_proto_opts \
        --splice=$splice --delta-order=$delta_order --dir=$dir \
        $num_in >$nnet_proto
      cnn_fea=$(cat $nnet_proto | grep -v '^$' | tail -n1 | awk '{ print $5; }')
      utils/nnet/make_nnet_proto.py $proto_opts \
        --no-smaller-input-weights \
        ${bn_dim:+ --bottleneck-dim=$bn_dim} \
        "$cnn_fea" $num_out $hid_layers $hid_dim >>$nnet_proto
      ;;
    lstm)
      utils/nnet/make_lstm_proto.py $proto_opts \
        $num_in $num_out >$nnet_proto
      ;;
    blstm)
      utils/nnet/make_blstm_proto.py $proto_opts \
        $num_in $num_out >$nnet_proto
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
  ${feature_transform:+ --output-feature-transform "$feature_transform"} \
  ${split_feats:+ --split-feats $split_feats} \
  --learn-rate $learn_rate \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${utt_weights:+ --utt-weights "$utt_weights"} \
  ${config:+ --config $config} \
  $nnet_init "$infeats_tr" "$infeats_cv" "$feats_tr" "$feats_cv" $dir

echo "$0: Successfuly finished. '$dir'"

sleep 3
exit 0
