#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# MMI training (or optionally boosted MMI, if you give the --boost option).
# 4 iterations (by default) of Extended Baum-Welch update.
#
# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.

# Begin configuration section.
cmd=run.pl
num_iters=4
boost=0.0
cancel=true # if true, cancel num and den counts on each frame.
tau=400
weight_tau=10
acwt=0.1
stage=0
smooth_to_mode=true
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: steps/train_mmi.sh <data> <lang> <ali> <denlats> <exp>"
  echo " e.g.: steps/train_mmi.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
dir=$5
mkdir -p $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

for f in $data/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;

sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

# Set up features

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f $alidir/trans.1 ] && echo Using transforms from $alidir && \
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi


cur_mdl=$alidir/final.mdl
x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of MPE training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-rescore-lattice $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-mpe-post --acoustic-scale=$acwt $cur_mdl \
        "ark,s,cs:gunzip -c $alidir/ali.JOB.gz |" ark:- ark:- \| \
      gmm-acc-stats2 $cur_mdl "$feats" ark,s,cs:- \
        $dir/num_acc.$x.JOB.acc $dir/den_acc.$x.JOB.acc || exit 1;

    n=`echo $dir/{num,den}_acc.$x.*.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.log \
      gmm-sum-accs $dir/den_acc.$x.acc $dir/den_acc.$x.*.acc || exit 1;
    rm $dir/den_acc.$x.*.acc
    $cmd $dir/log/num_acc_sum.$x.log \
      gmm-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
    rm $dir/num_acc.$x.*.acc

    # note: this tau value is for smoothing towards model parameters, not
    # as in the Boosted MMI paper, not towards the ML stats as in the earlier
    # work on discriminative training (e.g. my thesis).  
    # You could use gmm-ismooth-stats to smooth to the ML stats, if you had
    # them available [here they're not available if cancel=true].
    if ! $smooth_to_model; then
      echo "Iteration $x of MPE: computing ml (smoothing) stats"
      $cmd JOB=1:$nj $dir/log/acc_ml.$x.JOB.log \
        gmm-acc-stats $cur_mdl "$feats" \
          "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" \
          $dir/ml.$x.JOB.acc || exit 1;
      $cmd $dir/log/acc_ml_sum.$x.log \
        gmm-sum-accs $dir/ml.$x.acc $dir/ml.$x.*.acc || exit 1;
      rm $dir/ml.$x.*.acc
      num_stats="gmm-ismooth-stats --tau=$tau $dir/ml.$x.acc $dir/num_acc.$x.acc -|"
    else 
      num_stats="gmm-ismooth-stats --smooth-from-model=true --tau=$tau $cur_mdl $dir/num_acc.$x.acc -|"
    fi  
    
    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw $cur_mdl "$num_stats" $dir/den_acc.$x.acc - \| \
      gmm-est-weights-ebw - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
    rm $dir/{den,num}_acc.$x.acc
  fi
  cur_mdl=$dir/$[$x+1].mdl

  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.

 tail -n 50 $dir/log/acc.$x.*.log | perl -e 'while(<STDIN>) { if(m/lattice-to-mpe-post.+Overall average frame-accuracy is (\S+) over (\S+) frames/) { $tot_objf += $1*$2; $tot_frames += $2; }} $tot_objf /= $tot_frames; print "$tot_objf $tot_frames\n"; ' > $dir/tmpf
  objf=`cat $dir/tmpf | awk '{print $1}'`;
  nf=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf
  impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
  impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
  # This gives us a projected objective function improvement.
  echo "Iteration $x: objf was $objf, MPE auxf change was $impr" | tee $dir/objf.$x.log
  x=$[$x+1]
done

echo "MPE training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;
