#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# MMI training (or optionally boosted MMI, if you give the --boost option),
# for SGMMs.  4 iterations (by default) of Extended Baum-Welch update.
#
# Begin configuration section.
cmd=run.pl
num_iters=4
boost=0.0
cancel=true # if true, cancel num and den counts on each frame.
acwt=0.1
stage=0
update_opts=
transform_dir=
onlyOneJob=
onlyOneIter=
skipEBW=
skipACC=
# End configuration section

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: steps/train_mmi_sgmm2.sh <data> <lang> <ali> <denlats> <exp>"
  echo " e.g.: steps/train_mmi_sgmm2.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."  
  echo "  --transform-dir <transform-dir>                  # directory to find fMLLR transforms."
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
dir=$5
mkdir -p $dir/log

for f in $data/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;

sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

# Set up featuresl

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" \
    && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
else
  echo "$0: no fMLLR transforms."
fi

if [ -f $alidir/vecs.1 ]; then
  echo "$0: using speaker vectors from $alidir"
  spkvecs_opt="--spk-vecs=ark:$alidir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk"
else
  echo "$0: no speaker vectors."
  spkvecs_opt=
fi

if [ -f $alidir/gselect.1.gz ]; then
  echo "$0: using Gaussian-selection info from $alidir"
  gselect_opt="--gselect=ark:gunzip -c $alidir/gselect.JOB.gz|"
else
  echo "$0: error: no Gaussian-selection info found" && exit 1;
fi

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi


cur_mdl=$alidir/final.mdl
x=0
if [ "$onlyOneIter" ]; then
    # Just one iteration, instead of going from x=0 to $num_iters
    if (($onlyOneIter > $x)); then
	cur_mdl=$dir/$x.mdl
    fi
    x=$onlyOneIter
    num_iters=$[$x+1]
    echo "Performing only one iteration (iter=$x) with initial model $cur_mdl"
fi
while [ $x -lt $num_iters ]; do
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
      echo "Starting iteration $x of MMI training on" `date`
      if [ ! $skipACC ]; then
	  if [ "$onlyOneJob" ]; then
              # Just only this one job instead of JOB=1:$nj
	      echo "Starting job $onlyOneJob of iteration $x"
	      $cmd JOB=$onlyOneJob $dir/log/acc.$x.JOB.log \
		  sgmm2-rescore-lattice "$gselect_opt" $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
		  lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
		  sum-post --merge=$cancel --scale1=-1 \
		  ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
		  sgmm2-acc-stats2 "$gselect_opt" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
		  $dir/num_acc.$x.JOB.acc $dir/den_acc.$x.JOB.acc || exit 1;
	  else
	      echo "Starting jobs 1 to $nj of iteration $x"
	      $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
		  sgmm2-rescore-lattice "$gselect_opt" $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
		  lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
		  sum-post --merge=$cancel --scale1=-1 \
		  ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
		  sgmm2-acc-stats2 "$gselect_opt" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
		  $dir/num_acc.$x.JOB.acc $dir/den_acc.$x.JOB.acc || exit 1;
	  fi
      fi
      if [ ! $skipEBW ]; then
	  echo "Starting EBW update for iteration $x"
	  n=`echo $dir/{num,den}_acc.$x.*.acc | wc -w`;
	  [ "$n" -ne $[$nj*2] ] && \
	      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
	  $cmd $dir/log/den_acc_sum.$x.log \
	      sgmm2-sum-accs $dir/den_acc.$x.acc $dir/den_acc.$x.*.acc || exit 1;
	  rm $dir/den_acc.$x.*.acc
	  $cmd $dir/log/num_acc_sum.$x.log \
	      sgmm2-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
	  rm $dir/num_acc.$x.*.acc
	  
	  $cmd $dir/log/update.$x.log \
	      sgmm2-est-ebw $update_opts $cur_mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
      fi
  else
      echo "Skipping iteration $x of MMI training (already completed)"
  fi
  cur_mdl=$dir/$[$x+1].mdl

  if [ ! $skipEBW ]; then
  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.  Note: this code is same as in train_mmi.sh
      tail -n 50 $dir/log/acc.$x.*.log | perl -e '$acwt=shift @ARGV; while(<STDIN>) { if(m/sgmm2-acc-stats2.+Overall weighted acoustic likelihood per frame was (\S+) over (\S+) frames/) { $tot_aclike += $1*$2; $tot_frames1 += $2; } if(m|lattice-to-post.+Overall average log-like/frame is (\S+) over (\S+) frames.  Average acoustic like/frame is (\S+)|) { $tot_den_lat_like += $1*$2; $tot_frames2 += $2; $tot_den_aclike += $3*$2; } } if (abs($tot_frames1 - $tot_frames2) > 0.01*($tot_frames1 + $tot_frames2)) { print STDERR "Frame-counts disagree $tot_frames1 versus $tot_frames2\n"; } $tot_den_lat_like /= $tot_frames2; $tot_den_aclike /= $tot_frames2; $tot_aclike *= ($acwt / $tot_frames1);  $num_like = $tot_aclike + $tot_den_aclike; $per_frame_objf = $num_like - $tot_den_lat_like; print "$per_frame_objf $tot_frames1\n"; ' $acwt > $dir/tmpf
      objf=`cat $dir/tmpf | awk '{print $1}'`;
      nf=`cat $dir/tmpf | awk '{print $2}'`;
      rm $dir/tmpf
      impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
      impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
      echo "Iteration $x: objf was $objf, MMI auxf change was $impr" | tee $dir/objf.$x.log
  fi
  x=$[$x+1]
done

echo "MMI training finished: Job=$onlyOneJob iter=$onlyOneIter skipACC=$skipACC skipEBW=$skipEBW"

if [ ! $skipEBW ]; then
    rm $dir/final.mdl 2>/dev/null
    ln -s $x.mdl $dir/final.mdl
fi

exit 0;
