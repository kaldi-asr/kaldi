#!/bin/bash

# Copyright 2012/2013  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
feature_transform= # select feature transform (=splice,rescaling,...) (don't build new one)
#
model_size=8000000 # nr. of parameteres in MLP
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
bn_dim=            # set a value to get a bottleneck network
hid_dim=           # select hidden dimension directly (override $model_size)
dbn=               # select DBN to prepend to the MLP initialization
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
copy_feats=true  # resave the train features in the re-shuffled order to tmpdir
# feature config (applies always)
apply_cmvn=false # apply normalization to input features?
 norm_vars=false # use variance normalization?
delta_order=
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

# LABELS
labels=            # use these labels to train (override deafault pdf alignments) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_opts=        # options, passed to the training script
train_tool=        # optionally change the training tool

# OTHER
use_gpu_id= # manually select GPU id to run on, (-1 disables GPU)
analyze_alignments=true # run the alignment analysis script
seed=777    # seed value used for training data shuffling and initialization
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

silphonelist=`cat $lang/phones/silence.csl` || exit 1;


for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"

mkdir -p $dir/{log,nnet}

#skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z $labels ]; then
  echo "Using targets '$labels' (by force)"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  #define pdf-alignment rspecifiers
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  if [[ "$alidir" == "$alidir_cv" ]]; then
    labels="$labels_tr"
  else
    labels="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz $alidir_cv/ali.*.gz |\" ark:- |"
  fi

  #get the priors, get pdf-counts from alignments
  analyze-counts --binary=false "$labels_tr" $dir/ali_train_pdf.counts || exit 1
  #copy the old transition model, will be needed by decoder
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
  #copy the tree
  cp $alidir/tree $dir/tree || exit 1

  #analyze the train/cv alignments
  if [ "$analyze_alignments" == "true" ]; then
    utils/nnet/analyze_alignments.sh "TRAINING SET" "ark:gunzip -c $alidir/ali.*.gz |" $dir/final.mdl $lang > $dir/__ali_stats_train
    utils/nnet/analyze_alignments.sh "VALIDATION SET" "ark:gunzip -c $alidir_cv/ali.*.gz |" $dir/final.mdl $lang > $dir/__ali_stats_cv
  fi
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

#re-save the shuffled features, so they are stored sequentially on the disk in /tmp/
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d); mv $dir/train.scp $dir/train.scp_non_local
  utils/nnet/copy_feats.sh $dir/train.scp_non_local $tmpdir $dir/train.scp
  #remove data on exit...
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; rm -r $tmpdir" EXIT
fi

#create a 10k utt subset for global cmvn estimates
head -n 10000 $dir/train.scp > $dir/train.scp.10k



###### PREPARE FEATURE PIPELINE ######

#read the features
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"

#optionally add per-speaker CMVN
if [ $apply_cmvn == "true" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Cannot find cmvn stats $data/cmvn.scp" && exit 1;
  [ ! -r $data_cv/cmvn.scp ] && echo "Cannot find cmvn stats $data_cv/cmvn.scp" && exit 1;
  feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
  feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
  # keep track of norm_vars option
  echo "$norm_vars" >$dir/norm_vars 
else
  echo "apply_cmvn is disabled (per speaker norm. on input features)"
fi

#optionally add deltas
if [ "$delta_order" != "" ]; then
  feats_tr="$feats_tr add-deltas --delta-order=$delta_order ark:- ark:- |"
  feats_cv="$feats_cv add-deltas --delta-order=$delta_order ark:- ark:- |"
  echo "$delta_order" > $dir/delta_order
  echo "add-deltas (delta_order $delta_order)"
fi

#get feature dim
echo "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"

# Now we will start building complex feature_transform which will 
# be forwarded in CUDA to gain more speed.
#
# We will use 1GPU for both feature_transform and MLP training in one binary tool. 
# This is against the kaldi spirit, but it is necessary, because on some sites a GPU 
# cannot be shared accross by two or more processes (compute exclusive mode),
# and we would like to use single GPU per training instance,
# so that the grid resources can be used efficiently...

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
      [ -z $transf ] && $alidir/final.mat
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
  nnet-forward ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} \
    $feature_transform_old "$(echo $feats_tr | sed 's|train.scp|train.scp.10k|')" \
    ark:- 2>$dir/log/nnet-forward-cmvn.log |\
  compute-cmvn-stats ark:- - | cmvn-to-nnet - - |\
  nnet-concat --binary=false $feature_transform_old - $feature_transform
fi


###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
(cd $dir; [ ! -f final.feature_transform ] && ln -s $(basename $feature_transform) final.feature_transform )


###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
if [ ! -z "$mlp_init" ]; then
  echo "Using pre-initalized network $mlp_init";
else
  echo "Getting input/output dims :"
  #initializing the MLP, get the i/o dims...
  #input-dim
  num_fea=$(feat-to-dim "$feats_tr nnet-forward $feature_transform ark:- ark:- |" - )
  { #optioanlly take output dim of DBN
    [ ! -z $dbn ] && num_fea=$(nnet-forward "nnet-concat $feature_transform $dbn -|" "$feats_tr" ark:- | feat-to-dim ark:- -)
    [ -z "$num_fea" ] && echo "Getting nnet input dimension failed!!" && exit 1
  }

  #output-dim
  [ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  #run the MLP initializing script
  mlp_init=$dir/nnet.init
  utils/nnet/init_nnet.sh --model_size $model_size --hid_layers $hid_layers \
    ${bn_dim:+ --bn-dim $bn_dim} \
    ${hid_dim:+ --hid-dim $hid_dim} \
    --seed $seed ${init_opts} \
    ${config:+ --config $config} \
    $num_fea $num_tgt $mlp_init || exit 1

  #optionally prepend dbn to the initialization
  if [ ! -z $dbn ]; then
    mlp_init_old=$mlp_init; mlp_init=$dir/nnet_$(basename $dbn)_dnn.init
    nnet-concat $dbn $mlp_init_old $mlp_init 
  fi
fi


###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/train_nnet_scheduler.sh \
  --feature-transform $feature_transform \
  --learn-rate $learn_rate \
  --seed $seed \
  ${train_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${config:+ --config $config} \
  ${use_gpu_id:+ --use-gpu-id $use_gpu_id} \
  $mlp_init "$feats_tr" "$feats_cv" "$labels" $dir || exit 1


echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
