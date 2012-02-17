#!/bin/bash

# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

dir=exp/nnet-bn_tri3a_s4b_net
mkdir -p $dir/{log,nnet}


#source directory with GMM models
srcdir=exp/tri3a
srcmodel=$srcdir/final.mdl




###### SELECT FEATURES ######
# This one uses all the SI-284 data. 
cat data/train.scp | scripts/shuffle_list.pl ${seed:-777} > $dir/train.scp

N=$(cat $dir/train.scp | wc -l)
CV=$((N/10))

head -n $((N-CV)) $dir/train.scp > $dir/train.scp.tr
tail -n $CV $dir/train.scp > $dir/train.scp.cv

#splice the raw MFCC features
feats="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:$dir/train.scp ark:- |"
feats_tr="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:$dir/train.scp.tr ark:- |"
feats_cv="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:$dir/train.scp.cv ark:- |"
#features  for LDA computation (same order as alignments from tri3a)
rawfeats="ark:splice-feats --print-args=false --left-context=15 --right-context=15 scp:data/train.scp ark:- |"

#compute LDA
silphonelist=`cat data/silphones.csl`
if [ ! -e $dir/lda.mat ]; then
  # Now accumulate LDA stats.
  echo lda stats
  [ -f $dir/.error ] && rm $dir/.error
  (for n in 1 2 3; do
    ali-to-post "ark:gunzip -c $srcdir/cur$n.ali.gz|" ark:- 
  done | weight-silence-post 0.0 $silphonelist $srcmodel ark:- ark:- | \
  acc-lda $srcmodel "${rawfeats}" ark:- $dir/lda.acc ) 2>$dir/lda_acc.log || touch $dir/.error 

  [ -f $dir/.error ] &&  echo lda-acc error && exit 1
  # Compute the transform
  echo lda estimation
  est-lda --dim=240 $dir/lda.mat $dir/lda.acc  2>$dir/lda_est.log || exit 1
fi

#add transform to feats
feats="$feats transform-feats $dir/lda.mat ark:- ark:- |"
feats_tr="$feats_tr transform-feats $dir/lda.mat ark:- ark:- |"
feats_cv="$feats_cv transform-feats $dir/lda.mat ark:- ark:- |"



###### SELECT ALIGNMENTS ######
#choose directory with alignments
dir_ali=exp/tri3a
echo alignments: $dir_ali
if [ ! -f $dir/cur.pdf ]; then
  #merge the alignment files
  for ii in 1 2 3; do
    gunzip -c $dir_ali/cur$ii.ali.gz
  done | gzip -c > $dir/cur.ali.gz
  #convert ali to pdf
  ali-to-pdf $dir_ali/final.mdl "ark:gunzip -c $dir/cur.ali.gz|" t,ark:$dir/cur.pdf
fi
labels="ark:$dir/cur.pdf"
#count the class frames for division by prior
scripts/count_class_frames.awk $dir/cur.pdf $dir/cur.counts



###### NORMALIZE FEATURES ######
#compute per-utterance CMN
cmn="ark:$dir/cmn.ark"
[ -f ${cmn#ark:} ] || compute-cmvn-stats "$feats" $cmn
feats="$feats apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=false $cmn ark:- ark:- |"


#compute global CVN
cvn="$dir/global_cvn.mat"
[ -f $cvn ] || compute-cmvn-stats "$feats" $cvn 
#add global CVN to feature extration
feats="$feats apply-cmvn --print-args=false --norm-vars=true $cvn ark:- ark:- |"
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=true $cvn ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=true $cvn ark:- ark:- |"


###### INITIALIZE THE NNET ######
mlp_init=$dir/nnet.init
num_tgt=$(gmm-copy --binary=false $dir_ali/final.mdl - | grep NUMPDFS | awk '{ print $4 }')
scripts/gen_mlp_init.py --dim=240:800:30:800:${num_tgt} --gauss --negbias --seed=777 --linBNdim=30 > $mlp_init



########################################### 
# TRAINING CONFIG ######
########################################### 
#global config for trainig
max_iters=20
start_halving_inc=0.5
end_halving_inc=0.1
lrate=0.00005



nnet-train-xent-hardlab-perutt --cross-validate=true $mlp_init "$feats_cv" "$labels" &> $dir/log/prerun.log
if [ $? != 0 ]; then cat $dir/log/prerun.log; exit 1; fi
acc=$(cat $dir/log/prerun.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
echo CROSSVAL PRERUN ACCURACY $acc

mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
halving=0
for iter in $(seq -w $max_iters); do
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  nnet-train-xent-hardlab-perutt --learn-rate=$lrate $mlp_best "$feats_tr" "$labels" $mlp_next &> $dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi
  tr_acc=$(cat $dir/log/iter$iter.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
  echo TRAIN ITERATION $iter ACCURACY $tr_acc LRATE $lrate
  nnet-train-xent-hardlab-perutt --cross-validate=true $mlp_next "$feats_cv" "$labels" 1>>$dir/log/iter$iter.log 2>>$dir/log/iter$iter.log
  if [ $? != 0 ]; then cat $dir/log/iter$iter.log; exit 1; fi

  #accept or reject new parameters
  acc_new=$(cat $dir/log/iter$iter.log | grep FRAME_ACCURACY | tail -n 1 | cut -d' ' -f 3 | cut -d'%' -f 1)
  echo CROSSVAL ITERATION $iter ACCURACY $acc_new
  acc_prev=$acc
  if [ 1 == $(awk 'BEGIN{print('$acc_new' > '$acc')}') ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/$mlp_base.iter${iter}_lr$(printf "%.4g" $lrate)_tr$(printf "%.5g" $tr_acc)_cv$(printf "%.5g" $acc_new)
    mv $mlp_next $mlp_best
    echo nnet $mlp_best accepted
  else
    mlp_reject=$dir/nnet/$mlp_base.iter${iter}_lr$(printf "%.4g" $lrate)_tr$(printf "%.5g" $tr_acc)_cv$(printf "%.5g" $acc_new)
    mv $mlp_next $mlp_reject
    echo nnet $mlp_reject rejected 
  fi

  #stopping criterion
  if [[ 1 == $halving && 1 == $(awk 'BEGIN{print('$acc' < '$acc_prev'+'$end_halving_inc')}') ]]; then
    echo finished, too small improvement $(awk 'BEGIN{print('$acc'-'$acc_prev')}')
    break
  fi

  #start annealing when improvement is low
  if [ 1 == $(awk 'BEGIN{print('$acc' < '$acc_prev'+'$start_halving_inc')}') ]; then
    halving=1
  fi
  
  #do annealing
  if [ 1 == $halving ]; then
    lrate=$(awk 'BEGIN{print('$lrate'*0.5)}')
  fi
done

if [ $mlp_best != $mlp_init ]; then 
  iter=$(echo $mlp_best | sed 's/^.*iter\([0-9][0-9]*\).*$/\1/')
fi
mlp_final=$dir/${mlp_base}_final_iter${iter:-0}_acc${acc}
ln -s $PWD/$mlp_best $mlp_final

if [ -e $dir/final.nnet ]; then
  unlink $dir/final.nnet
fi
ln -s $PWD/$mlp_final $dir/final.nnet

echo final network $mlp_final

