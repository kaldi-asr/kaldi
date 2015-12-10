#!/bin/bash

# Copyright 2015 Sri Harish Mallidi
# Apache 2.0

# Begin configuration section. 
nj=4
cmd=run.pl

remove_last_components=4 # remove N last components from the nnet
nnet_fwdpass_opts=
nnet_fwdpass_tool=theano-nnet/nnet1_v2/nn_fwdpass.py

remove_last_components=4 # remove N last components from the nnet
                         # LDA takes outputs from remaining neuralnet

# LDA related
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
lda_dim=300

nj=4
cmd=run.pl
use_gpu="no" # yes|no|optionaly
stage=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <data-train> <lang-dir> <ali-train> <nnet-dir> <lda-dir>"
   echo "e.g.: $0 data/train data/lang exp/tri3a_ali exp/dnn4a some_dir/lda"
   echo ""
   echo "This scripts does lda on MLP activations. Useful for doing LDA on Bottleneck, HATS, TANDEM."
   echo "Creats lda-dir in nnet-dir format, so lda features can be computed using"
   echo "steps/nnet/make_bn_feats.sh"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --nnetdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   exit 1;
fi

data=$1
lang=$2
alidir=$3
nnetdir=$4
ldadir=$5

sdata=$data/split$nj;

logdir=$ldadir/log; mkdir -p $logdir 

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $ldadir/num_jobs

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nnet=$nnetdir/final_nnet.pklz
feat_preprocess=$nnetdir/feat_preprocess.pkl
model=$nnetdir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $nnet $model $feat_preprocess; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# Concat feature transform with trimmed MLP:
cp $nnetdir/feat_preprocess.pkl $ldadir/feat_preprocess.pkl
nnet=$ldadir/feature_extractor_nnet.pklz
# Create trimmed MLP:
python theano-nnet/nnet1_v2/nnet_copy.py \
  --remove-last-components=$remove_last_components \
  $nnetdir/final_nnet.pklz $nnet 2>$logdir/feature_extractor.log || exit 1
(cd $ldadir; ln -sf feature_extractor_nnet.pklz final_nnet.pklz; cd -) #for nn_fwdpass.py

dir=$ldadir
transf=$dir/lda$lda_dim.mat

if [ $stage -le 0 ]; then
echo "Getting posts ..."
ali-to-post "ark:gunzip -c $alidir/ali.*.gz|" ark:- | weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:$ldadir/post.ark || exit 1;

echo "Accumulating LDA statistics."
$cmd JOB=1:$nj $logdir/lda_acc.JOB.log \
  $nnet_fwdpass_tool $nnet_fwdpass_opts \
    --feat-preprocess=$ldadir/feat_preprocess.pkl \
    --utt2spk-file=$data/utt2spk --cmvn-scp=$data/cmvn.scp \
    $ldadir $sdata/JOB/ \| \
    acc-lda --rand-prune=$randprune $alidir/final.mdl ark,t:- ark:$ldadir/post.ark $ldadir/lda.JOB.acc || exit 1;
fi

if [ $stage -le 1 ]; then
echo "Estimating LDA matrix."
run.pl $logdir/lda_est.log \
  est-lda --write-full-matrix=$dir/full.mat --dim=$lda_dim $transf $dir/lda.*.acc || exit 1;
rm $dir/lda.*.acc
fi

(cd $ldadir/; ln -s lda$lda_dim.mat lda.mat; cd - )

echo "$0 successfuly finished.. $ldadir/lda.mat"

sleep 3
exit 0;

