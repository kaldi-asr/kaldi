#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#                 Korbinian Riedhammer

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
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/train_tandem_mmi.sh <data1> <data2> <lang> <ali> <denlats> <exp>"
  echo " e.g.: steps/train_tandem_mmi.sh {mfcc,bottleneck}/data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  
  exit 1;
fi

data1=$1
data2=$2
lang=$3
alidir=$4
denlatdir=$5
dir=$6

mkdir -p $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

for f in $data1/feats.scp $data2/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
cp $alidir/{final.mdl,tree} $dir
silphonelist=`cat $lang/phones/silence.csl` || exit 1;


# Set up features

sdata1=$data1/split$nj
sdata2=$data2/split$nj
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $alidir/normft2 2>/dev/null`

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

case $feat_type in
  delta) 
  	echo "$0: feature type is $feat_type"
  	;;
  lda) 
  	echo "$0: feature type is $feat_type"
    cp $alidir/{lda,final}.mat $dir/ || exit 1; 
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features, 
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $alidir/{splice_opts,normft2,tandem} $dir 2>/dev/null

if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
##

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi


cur_mdl=$alidir/final.mdl
x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of MMI training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-rescore-lattice $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
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

    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw --tau=$tau $cur_mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc - \| \
      gmm-est-weights-ebw --weight-tau=$weight_tau - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
    rm $dir/{den,num}_acc.$x.acc
  fi
  cur_mdl=$dir/$[$x+1].mdl

  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.

  tail -n 50 $dir/log/acc.$x.*.log | perl -e '$acwt=shift @ARGV; while(<STDIN>) { if(m/gmm-acc-stats2.+Overall weighted acoustic likelihood per frame was (\S+) over (\S+) frames/) { $tot_aclike += $1*$2; $tot_frames1 += $2; } if(m|lattice-to-post.+Overall average log-like/frame is (\S+) over (\S+) frames.  Average acoustic like/frame is (\S+)|) { $tot_den_lat_like += $1*$2; $tot_frames2 += $2; $tot_den_aclike += $3*$2; } } if (abs($tot_frames1 - $tot_frames2) > 0.01*($tot_frames1 + $tot_frames2)) { print STDERR "Frame-counts disagree $tot_frames1 versus $tot_frames2\n"; } $tot_den_lat_like /= $tot_frames2; $tot_den_aclike /= $tot_frames2; $tot_aclike *= ($acwt / $tot_frames1);  $num_like = $tot_aclike + $tot_den_aclike; $per_frame_objf = $num_like - $tot_den_lat_like; print "$per_frame_objf $tot_frames1\n"; ' $acwt > $dir/tmpf
  objf=`cat $dir/tmpf | awk '{print $1}'`;
  nf=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf
  impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
  impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
  echo "Iteration $x: objf was $objf, MMI auxf change was $impr" | tee $dir/objf.$x.log
  x=$[$x+1]
done

echo "MMI training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;
