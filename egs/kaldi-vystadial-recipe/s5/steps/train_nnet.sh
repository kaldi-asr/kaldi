#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Begin configuration.
cmd=run.pl

# nnet config
model_size=3000000 # nr. of parameteres in MLP
hid_layers=2      # nr. of hidden layers (prior to sotfmax or bottleneck)
bn_dim=           # set value to get a bottleneck network
hid_dim=          # set this to override the $model_size
mlp_init=         # set this to override MLP initialization
# training config
learn_rate=0.008  # initial learning rate
momentum=0.0      # momentum
l1_penalty=0.0     # L1 regualrization constant (lassoo)
l2_penalty=0.0     # L2 regualrization constant (weight decay)
# data processing config
bunch_size=256     # size of the training block
cache_size=16384   # size of the randimizatio cache
randomize=true    # do the frame level randomization
# feature config
norm_vars=false # normalize the FBANKs (CVN)
splice_lr=15    # temporal splicing
feat_type=traps
dct_basis=16    # nr. od DCT basis
# scheduling config
min_iters=    # set to enforce minimum number of iterations
max_iters=20  # maximum number of iterations
start_halving_inc=0.5 # frm-accuracy improvement to begin learn_rate reduction
end_halving_inc=0.1   # frm-accuracy improvement to terminate the training
halving_factor=0.5    # factor to multiply learn_rate
# tool config
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff" # training tool used for training / cross validation
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz $data/feats.scp $data_cv/feats.scp $data/cmvn.scp $data_cv/cmvn.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo "$0 [info]: Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"

mkdir -p $dir/{log,nnet}

###### PREPARE ALIGNMENTS ######
echo "Preparing alignments"
#convert ali to pdf
labels_tr="ark:$dir/ali_train.pdf"
ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" $labels_tr 2> $dir/ali_train.pdf_log || exit 1
if [[ "$alidir" == "$alidir_cv" ]]; then
  labels=$labels_tr
else
  #convert ali to pdf (cv set)
  labels_cv="ark:$dir/ali_cv.pdf"
  ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir_cv/ali.*.gz |" $labels_cv 2> $dir/ali_cv.pdf_log || exit 1
  #merge the two parts (scheduler expects one file in $labels)
  labels="ark:$dir/ali_train_and_cv.pdf"
  cat $dir/ali_train.pdf $dir/ali_cv.pdf > $dir/ali_train_and_cv.pdf
fi

#get the priors, count the class examples from alignments
analyze-counts --binary=false $labels_tr $dir/ali_train.counts 2>$dir/ali_train.counts_log || exit 1
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl 2>$dir/final.mdl_log || exit 1
cp $alidir/tree $dir/tree || exit 1

#analyze the train/cv alignments
utils/nnet/analyze_alignments.sh "TRAINING SET" "ark:gunzip -c $alidir/ali.*.gz |" $dir/final.mdl $lang > $dir/__ali_stats_train
utils/nnet/analyze_alignments.sh "VALIDATION SET" "ark:gunzip -c $alidir_cv/ali.*.gz |" $dir/final.mdl $lang > $dir/__ali_stats_cv


###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | utils/shuffle_list.pl ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

#get feature dim
echo -n "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false scp:$dir/train.scp -)
echo $feat_dim

#add per-speaker CMVN
echo "Will use CMVN statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
cmvn="scp:$data/cmvn.scp"
cmvn_cv="scp:$data_cv/cmvn.scp"
feats_tr="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk $cmvn scp:$dir/train.scp ark:- |"
feats_cv="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk $cmvn_cv scp:$dir/cv.scp ark:- |"
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars 

#add splicing
splice_opts="--left-context=$splice_lr --right-context=$splice_lr"
feats_tr="$feats_tr splice-feats --print-args=false $splice_opts ark:- ark:- |"
feats_cv="$feats_cv splice-feats --print-args=false $splice_opts ark:- ark:- |"
# keep track of splice_opts
echo "$splice_opts" >$dir/splice_opts

#choose further processing of spliced features
echo "Feature type : $feat_type"
case $feat_type in
  plain)
  ;;
  traps)
    #generate hamming+dct transform
    transf=$dir/hamm_dct.mat
    echo "Preparing Hamming DCT transform : $transf"
    utils/nnet/gen_hamm_mat.py --fea-dim=$feat_dim --splice=$splice_lr > $dir/hamm.mat
    utils/nnet/gen_dct_mat.py --fea-dim=$feat_dim --splice=$splice_lr --dct-basis=$dct_basis > $dir/dct.mat
    compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat $transf 2>${transf}_log || exit 1
    #convert transform to NNET format
    {
      echo "<biasedlinearity> $((feat_dim*dct_basis)) $((feat_dim*(2*splice_lr+1)))"
      cat $transf
      echo -n ' [ '
      for i in $(seq $((feat_dim*dct_basis))); do echo -n '0 '; done
      echo ']'
    } > $transf.net
    #append transform to features
    feats_tr="$feats_tr nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
    feats_cv="$feats_cv nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
  ;;
  transf)
    transf=$dir/final.mat
    [ ! -f $alidir/final.mat ] && echo "Missing transform $alidir/final.mat" && exit 1;
    cp $alidir/final.mat $transf
    echo "Copied transform $transf"
    feats_tr="$feats_tr transform-feats $transf ark:- ark:- |"
    feats_cv="$feats_cv transform-feats $transf ark:- ark:- |"
  ;;
  transf-sat)
    echo yet unimplemented...
    exit 1;
  ;;
  *)
    echo "Unknown feature type $feat_type"
    exit 1;
  ;;
esac
# keep track of feat_type
echo $feat_type > $dir/feat_type

#renormalize the MLP input to zero mean and unit variance
cmvn_g="$dir/cmvn_glob.mat"
echo "Renormalizing MLP input features by : $cmvn_g"
compute-cmvn-stats --binary=false "$feats_tr" $cmvn_g 2>${cmvn_g}_log || exit 1
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"


###### INITIALIZE THE NNET ######

if [ "" != "$mlp_init" ]; then
  echo "Using pre-initalized netwk $mlp_init";
else
  echo -n "Initializng MLP : "
  num_fea=$((feat_dim*dct_basis))
  num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')
  # What is the topology?
  if [ "" == "$bn_dim" ]; then #MLP w/o bottleneck
    case "$hid_layers" in
      1) #3-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          num_hid=$((model_size/(num_fea+num_tgt)))
        fi
        mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_tgt}.init
        echo " $mlp_init"
        utils/nnet/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_tgt} \
          --gauss --negbias --seed=777 > $mlp_init
        ;;
      2|3|4|5|6|7|8|9|10) #(>3)-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          a=$((hid_layers-1))
          b=$((num_fea+num_tgt))
          c=$((-model_size))
          num_hid=$(awk "BEGIN{ num_hid= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(num_hid) }") 
        fi
        mlp_init=$dir/nnet_${num_fea}
        dim_arg=${num_fea}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$num_hid
          dim_arg=${dim_arg}:${num_hid}
        done
        mlp_init=${mlp_init}_${num_tgt}.init
        dim_arg=${dim_arg}:${num_tgt}
        echo " $mlp_init"
        utils/nnet/gen_mlp_init.py --dim=${dim_arg} --gauss --negbias --seed=777 > $mlp_init
        ;;
      *)
        echo "Unsupported number of hidden layers $hid_layers"
        exit 1;
    esac
  else #bn-system
    num_bn=$bn_dim
    case "$hid_layers" in # ie. number of layers in front of bottleneck
      1) #5-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          num_hid=$((model_size/(num_fea+num_tgt+(2*num_bn))))
        fi
        mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_bn}_${num_hid}_${num_tgt}.init
        echo " $mlp_init"
        utils/nnet/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_bn}:${num_hid}:${num_tgt} --gauss --negbias --seed=777 --linBNdim=$num_bn > $mlp_init
        ;;
      2|3|4|5|6|7|8|9|10) #(>5)-layer MLP
        if [ "" != "$hid_dim" ]; then
          num_hid=$hid_dim
        else
          a=$((hid_layers-1))
          b=$((num_fea+2*num_bn+num_tgt))
          c=$((-model_size))
          num_hid=$(awk "BEGIN{ num_hid= -$b/(2*$a) + sqrt($b^2 -4*$a*$c)/(2*$a); print int(num_hid) }") 
        fi
        mlp_init=$dir/nnet_${num_fea}
        dim_arg=${num_fea}
        for i in $(seq $hid_layers); do
          mlp_init=${mlp_init}_$num_hid
          dim_arg=${dim_arg}:${num_hid}
        done
        mlp_init=${mlp_init}_${num_bn}lin_${num_hid}_${num_tgt}.init
        dim_arg=${dim_arg}:${num_bn}:${num_hid}:${num_tgt}
        echo " $mlp_init"
        utils/nnet/gen_mlp_init.py --dim=${dim_arg} --gauss --negbias --seed=777 --linBNdim=$num_bn > $mlp_init
        ;;
      *)
        echo "Unsupported number of hidden layers $hid_layers"
        exit 1;
    esac
  fi
fi



###### TRAIN ######
echo "Starting training : "
source utils/nnet/train_nnet_scheduler.sh
echo "Training finished."
echo
if [ "" == "$mlp_final" ]; then
  echo "No final network returned!";
  exit 1;
else
  ( cd $dir; ln -s nnet/${mlp_final##*/} final.nnet; )
  echo "Final network $mlp_final linked to $dir/final.nnet";
fi

echo "Succeeded training the Neural Network : $dir/final.nnet"



