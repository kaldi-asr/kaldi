#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Begin configuration.
cmd=run.pl

# nnet config
modelsize=3000000 # nr. of parameteres in MLP
learnrate=0.0002  # initial learning rate
momentum=0.0      # momentum
l1penalty=0.0     # L1 regualrization constant (lassoo)
l2penalty=0.0     # L2 regualrization constant (weight decay)
# data processing config
bunchsize=256     # size of the training block
cachesize=16384   # size of the randimizatio cache
randomize=true    # do the frame level randomization
# feature config
#fea_dim=23      # feature dimensionality
norm_vars=false # normalize the FBANKs (CVN)
splice_lr=15    # temporal splicing
dct_basis=16    # nr. od DCT basis
# scheduling config
max_iters=20  # maximum number of iterations
start_halving_inc=0.5 # frm-accuracy improvement to begin learnrate reduction
end_halving_inc=0.1   # frm-accuracy improvement to terminate the training
halving_factor=0.5    # factor to multiply learnrate
# tool config
TRAIN_TOOL="nnet-train-xent-hardlab-frmshuff" # training tool used for training / cross validation
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/train_dev_nnet4L.sh <data-train> <data-dev> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: steps/train_dev_nnet4L.sh data/train data/cv exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   exit 1;
fi

data=$1
data_cv=$2
alidir=$3
alidir_cv=$4
dir=$5

for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz $data/feats.scp $data_cv/feats.scp $data/cmvn.scp $data_cv/cmvn.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo "$0 [info]: training Neural Netowork in dir $dir with feats $data alignments $alidir, (cross-validation on $data_cv $alidir_cv)."

mkdir -p $dir/{log,nnet}

###### PREPARE ALIGNMENTS ######
echo "Preparing alignments"
#convert ali to pdf
labels_tr="ark:$dir/ali_train.pdf"
ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" t,$labels_tr 2> $dir/log/ali2pdf_tr.log || exit 1
#convert ali to pdf (cv set)
labels_cv="ark:$dir/ali_cv.pdf"
ali-to-pdf $alidir_cv/final.mdl "ark:gunzip -c $alidir_cv/ali.*.gz |" t,$labels_cv 2> $dir/log/ali2pdf_cv.log || exit 1
#merge the two parts (scheduler expects one file in $labels)
labels="ark:$dir/ali_train_cv.pdf"
cat $dir/ali_train.pdf $dir/ali_cv.pdf > $dir/ali_train_cv.pdf

#get the priors, count the class examples from alignments
pdf-to-counts ark:$dir/ali_train.pdf $dir/ali_train.counts
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl
cp $alidir/tree $dir/tree



###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | utils/shuffle_list.pl ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

#get feature dim
echo -n "Getting feature dim"
fea_dim=$(feat-to-dim scp:$dir/train.scp -)
echo $fea_dim

#compute per-speaker CMVN
echo "Recalling cepstral mean and variance statistics"
cmvn="scp:$data/cmvn.scp"
cmvn_cv="scp:$data_cv/cmvn.scp"
echo "$norm_vars" >$dir/norm_vars # keep track of norm_vars option
feats_tr="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk $cmvn scp:$dir/train.scp ark:- |"
feats_cv="ark:apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk $cmvn_cv scp:$dir/cv.scp ark:- |"

#add splicing
splice_opts="--left-context=$splice_lr --right-context=$splice_lr"
echo "$splice_opts" >$dir/splice_opts # keep track of frame-splicing options
feats_tr="$feats_tr splice-feats --print-args=false $splice_opts ark:- ark:- |"
feats_cv="$feats_cv splice-feats --print-args=false $splice_opts ark:- ark:- |"

#generate hamming+dct transform
echo "Preparing Hamming DCT transform"
transf=$dir/hamm_dct.mat
utils/nnet/gen_hamm_mat.py --fea-dim=$fea_dim --splice=$splice_lr > $dir/hamm.mat
utils/nnet/gen_dct_mat.py --fea-dim=$fea_dim --splice=$splice_lr --dct-basis=$dct_basis > $dir/dct.mat
compose-transforms --binary=false $dir/dct.mat $dir/hamm.mat $transf 2>$dir/log/hamm_dct.log || exit 1
#convert transform to NNET format
{
  echo "<biasedlinearity> $((fea_dim*dct_basis)) $((fea_dim*(2*splice_lr+1)))"
  cat $transf
  echo -n ' [ '
  for i in $(seq $((fea_dim*dct_basis))); do echo -n '0 '; done
  echo ']'
} > $transf.net
#append transform to features
feats_tr="$feats_tr nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"
feats_cv="$feats_cv nnet-forward --print-args=false --silent=true $transf.net ark:- ark:- |"

#renormalize the MLP input to zero mean and unit variance
echo "Renormalizing MLP input features"
cmvn_g="$dir/cmvn_glob.mat"
compute-cmvn-stats --binary=false "$feats_tr" $cmvn_g 2> $dir/log/cmvn_glob.log || exit 1
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cmvn_g ark:- ark:- |"



###### INITIALIZE THE NNET ######
echo -n "Initializng MLP: "
num_fea=$((fea_dim*dct_basis))
num_tgt=$(hmm-info $alidir/final.mdl | grep pdfs | awk '{ print $NF }')
num_hid=$(awk "BEGIN{ num_hid= -($num_fea+$num_tgt) / 2 + sqrt(($num_fea+$num_tgt)^2 + 4*$modelsize) / 2; print int(num_hid) }") # D=sqrt(b^2-4ac); x=(-b+/-D) / 2a
mlp_init=$dir/nnet_${num_fea}_${num_hid}_${num_hid}_${num_tgt}.init
echo " $mlp_init"
utils/nnet/gen_mlp_init.py --dim=${num_fea}:${num_hid}:${num_hid}:${num_tgt} --gauss --negbias --seed=777 > $mlp_init



###### TRAIN ######
echo "Starting training:"
source utils/nnet/train_nnet_scheduler.sh
echo "Training finished."
if [ "" == "$mlp_final" ]; then
  echo "No final network returned!";
  exit 1;
else
  ( cd $dir; ln -s nnet/${mlp_final##*/} final.nnet; )
  echo "Final network $mlp_final";
fi

echo "Succeeded training the Neural Network in $dir"



