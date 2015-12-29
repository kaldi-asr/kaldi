#!/bin/bash

# Copyright 2012/2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
mlp_proto=         # select network prototype (initialize it)
# proto_opts="--no-softmax --activation-type=<Tanh> --hid-bias-mean=0.0 --hid-bias-range=1.0 --param-stddev-factor=0.01"
proto_opts="--no-softmax --activation-type=<Tanh> --hid-bias-mean=0.0 --hid-bias-range=1.0"
                   # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
#
hid_layers=2       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=512        # select hidden dimension
bn_dim=            # set a value to get a bottleneck network
dbn=               # select DBN to prepend to the MLP initialization
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
copy_feats=true # resave the train/cv features into /tmp (disabled by default)
 copy_feats_tmproot= # tmproot for copy-feats (optional)
# feature config (applies always)
cmvn_opts=
delta_opts=
# feature_transform:
splice=5         # temporal splicing
splice_step=1    # stepsize of the splicing (1 == no gap between frames)
feat_type=plain
# feature config (applies to feat_type traps)
traps_dct_basis=11 # nr. od DCT basis (applies to `traps` feat_type, splice10 )
# feature config (applies to feat_type transf) (ie. LDA+MLLT, no fMLLR)
transf=
splice_after_transf=5
# feature config (applies to feat_type lda)
lda_dim=300        # LDA dimension (applies to `lda` feat_type)
alidir=            # Useful for aann on LDA features

# LABELS
labels=            # use these labels to train (override deafault pdf alignments, has to be in 'Posterior' format, see ali-to-post) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)

# TRAINING SCHEDULER
learn_rate=0.00001 # initial learning rate
train_opts=        # options, passed to the training script
train_tool="nnet-train-frmshuff --objective-function=mse"
                   # optionally change the training tool
frame_weights=     # per-frame weights for gradient weighting

# OTHER
seed=777    # seed value used for training data shuffling and initialization
skip_cuda_check=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 3 ]; then
   echo "Usage: $0 <data-train> <data-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv exp/aann"
   echo ""
   echo " Training data : <data-train>  (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev> (for learn-rate/model selection based on cross-entopy)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --apply-cmvn <bool>      # apply CMN"
   echo "  --norm-vars <bool>       # add CVN if CMN already active"
   echo "  --splice <N>             # concatenate input features"
   echo "  --feat-type <type>       # select type of input features"
   echo ""
   echo "  --mlp-proto <file>       # use this NN prototype"
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo "  --hid-layers <N>         # number of hidden layers"
   echo "  --hid-dim <N>            # width of hidden layers"
   echo "  --bn-dim <N>             # make bottle-neck network with bn-with N"
   echo ""
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo "  --copy-feats <bool>      # copy input features to /tmp (it's faster)"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
dir=$3

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

# check if CUDA is compiled in,
if ! $skip_cuda_check; then
  cuda-compiled || { echo 'CUDA was not compiled in, skipping! Check src/kaldi.mk and src/configure' && exit 1; }
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

# re-save the train/cv features to /tmp, reduces LAN traffic, avoids disk-seeks due to shuffled features
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d $copy_feats_tmproot); mv $dir/train.scp{,_non_local}; mv $dir/cv.scp{,_non_local}
  copy-feats scp:$dir/train.scp_non_local ark,scp:$tmpdir/train.ark,$dir/train.scp || exit 1
  copy-feats scp:$dir/cv.scp_non_local ark,scp:$tmpdir/cv.ark,$dir/cv.scp || exit 1
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
fi

#create a 10k utt subset for global cmvn estimates
head -n 10000 $dir/train.scp > $dir/train.scp.10k


###### PREPARE FEATURE PIPELINE ######

# optionally import feature setup from pre-training,
if [ ! -z $feature_transform ]; then
  D=$(dirname $feature_transform)
  [ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
  [ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
  [ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
  [ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
  echo "Imported config : cmvn_opts='$cmvn_opts' delta_opts='$delta_opts'"
fi

# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"
# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
else
  echo "apply-cmvn is not used"
fi
# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "add-deltas with $delta_opts"
fi

# keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts 
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
#

# get feature dim
echo "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"

# Now we will start building complex feature_transform which will 
# be forwarded in CUDA to have fast run-time.
#
# We will use 1GPU for both feature_transform and MLP training in one binary tool. 
# This is against the kaldi spirit to have many independent small processing units, 
# but it is necessary because of compute exclusive mode, where GPU cannot be shared
# by multiple processes.

if [ ! -z "$feature_transform" ]; then
  echo "Using pre-computed feature-transform : '$feature_transform'"
  tmp=$dir/$(basename $feature_transform) 
  cp $feature_transform $tmp; feature_transform=$tmp
else
  # Generate the splice transform
  echo "Using splice +/- $splice , step $splice_step"
  feature_transform=$dir/tr_splice$splice-$splice_step.nnet
  utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice --splice-step=$splice_step > $feature_transform

  # Choose further processing of spliced features
  echo "Feature type : $feat_type"
  case $feat_type in
    plain)
    ;;
    traps)
      #generate hamming+dct transform
      feature_transform_old=$feature_transform
      feature_transform=${feature_transform%.nnet}_hamm_dct${traps_dct_basis}.nnet
      echo "Preparing Hamming DCT transform into : $feature_transform"
      #prepare matrices with time-transposed hamming and dct
      utils/nnet/gen_hamm_mat.py --fea-dim=$feat_dim --splice=$splice > $dir/hamm.mat
      utils/nnet/gen_dct_mat.py --fea-dim=$feat_dim --splice=$splice --dct-basis=$traps_dct_basis > $dir/dct.mat
      #put everything together
      compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat - | \
        transf-to-nnet - - | \
        nnet-concat --binary=false $feature_transform_old - $feature_transform || exit 1
    ;;
    transf)
      feature_transform_old=$feature_transform
      feature_transform=${feature_transform%.nnet}_transf_splice${splice_after_transf}.nnet
      [ -z $transf ] && transf=$alidir/final.mat
      [ ! -f $transf ] && echo "Missing transf $transf" && exit 1
      feat_dim=$(feat-to-dim "$feats_tr nnet-forward 'nnet-concat $feature_transform_old \"transf-to-nnet $transf - |\" - |' ark:- ark:- |" -)
      nnet-concat --binary=false $feature_transform_old \
        "transf-to-nnet $transf - |" \
        "utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice_after_transf |" \
        $feature_transform || exit 1
    ;;
    lda)
      transf=$dir/lda$lda_dim.mat
      #get the LDA statistics
      if [ ! -r "$dir/lda.acc" ]; then
        echo "LDA: Converting alignments to posteriors $dir/lda_post.scp"
        ali-to-post "ark:gunzip -c $alidir/ali.*.gz|" ark:- | \
          weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark,scp:$dir/lda_post.ark,$dir/lda_post.scp 2>$dir/log/ali-to-post-lda.log || exit 1;
        echo "Accumulating LDA statistics $dir/lda.acc on top of spliced feats"
        acc-lda --rand-prune=4.0 $alidir/final.mdl "$feats_tr nnet-forward $feature_transform ark:- ark:- |" scp:$dir/lda_post.scp $dir/lda.acc 2>$dir/log/acc-lda.log || exit 1;
      else
        echo "LDA: Using pre-computed stats $dir/lda.acc"
      fi
      #estimate the transform  
      echo "Estimating LDA transform $dir/lda.mat from the statistics $dir/lda.acc"
      est-lda --write-full-matrix=$dir/lda.full.mat --dim=$lda_dim $transf $dir/lda.acc 2>$dir/log/lda.log || exit 1;
      #append the LDA matrix to feature_transform
      feature_transform_old=$feature_transform
      feature_transform=${feature_transform%.nnet}_lda${lda_dim}.nnet
      transf-to-nnet $transf - | \
        nnet-concat --binary=false $feature_transform_old - $feature_transform || exit 1
      #remove the temporary file
      rm $dir/lda_post.{ark,scp}
    ;;
    *)
      echo "Unknown feature type $feat_type"
      exit 1;
    ;;
  esac
  # keep track of feat_type
  echo $feat_type > $dir/feat_type

  # Renormalize the MLP input to zero mean and unit variance
  feature_transform_old=$feature_transform
  feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
  echo "Renormalizing MLP input features into $feature_transform"
  nnet-forward --use-gpu=yes \
    $feature_transform_old "$(echo $feats_tr | sed 's|train.scp|train.scp.10k|')" \
    ark:- 2>$dir/log/nnet-forward-cmvn.log |\
  compute-cmvn-stats ark:- - | cmvn-to-nnet - - |\
  nnet-concat --binary=false $feature_transform_old - $feature_transform
  [ ! -f $feature_transform ] && cat $dir/log/nnet-forward-cmvn.log && echo "Error: Global CMVN failed, was the CUDA GPU okay?" && echo && exit 1
fi


###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
(cd $dir; [ ! -f final.feature_transform ] && ln -s $(basename $feature_transform) final.feature_transform )

echo "PREPARING LABELS"

if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else 
  # labels_tr="ark:nnet-forward ${dir}/final.feature_transform $feats_tr ark:- | feat-to-post ark:- ark:- |"
  # labels_cv="ark:nnet-forward ${dir}/final.feature_transform $feats_cv ark:- | feat-to-post ark:- ark:- |"

  o_feats_tr=`echo $feats_tr | sed 's@ark:copy-feats@ark,o:copy-feats@g'`
  o_feats_cv=`echo $feats_cv | sed 's@ark:copy-feats@ark,o:copy-feats@g'`
  labels_tr="$o_feats_tr nnet-forward ${dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |"
  labels_cv="$o_feats_cv nnet-forward ${dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |"

fi

###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
[ ! -z "$mlp_init" ] && echo "Using pre-initialized network '$mlp_init'";
if [ ! -z "$mlp_proto" ]; then 
  echo "Initializing using network prototype '$mlp_proto'";
  mlp_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; } 
fi
if [[ -z "$mlp_init" && -z "$mlp_proto" ]]; then
  echo "Getting input/output dims :"
  #initializing the MLP, get the i/o dims...
  #input-dim
  num_fea=$(feat-to-dim "$feats_tr nnet-forward $feature_transform ark:- ark:- |" - )
  num_tgt=$num_fea

  { #optioanlly take output dim of DBN
    [ ! -z $dbn ] && num_fea=$(nnet-forward "nnet-concat $feature_transform $dbn -|" "$feats_tr" ark:- | feat-to-dim ark:- -)
    [ -z "$num_fea" ] && echo "Getting nnet input dimension failed!!" && exit 1
  }

  # make network prototype
  mlp_proto=$dir/nnet.proto  
  echo "Genrating network prototype $mlp_proto"

  # my.extras/utils/nnet/make_aann_proto.py --activation-type='<Tanh>' --no-softmax --hid-bias-mean=0 --hid-bias-range=1 --bottleneck-dim=$bn_dim $num_fea $num_tgt $hid_layers $hid_dim > $mlp_proto
  python my.extras/utils/nnet/make_aann_proto.py $proto_opts \
    ${bn_dim:+ --bottleneck-dim=$bn_dim} \
    $num_fea $num_tgt $hid_layers $hid_dim > $mlp_proto

  # initialize
  mlp_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  echo "Initializing $mlp_proto -> $mlp_init"
  nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; }

  # optionally prepend dbn to the initialization
  if [ ! -z $dbn ]; then
    mlp_init_old=$mlp_init; mlp_init=$dir/nnet_$(basename $dbn)_dnn.init
    nnet-concat $dbn $mlp_init_old $mlp_init || exit 1 
  fi
fi

###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet/train_scheduler.sh \
  --feature-transform $feature_transform \
  --learn-rate $learn_rate \
  ${train_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${config:+ --config $config} \
  $mlp_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir || exit 1

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
