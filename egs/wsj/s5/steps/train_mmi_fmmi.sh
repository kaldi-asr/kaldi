#!/bin/bash
# by Johns Hopkins University (Author: Daniel Povey), 2012.  Apache 2.0.

# This script does MMI discriminative training, including
# feature-space (like fMPE) and model-space components. 
# If you give the --boost option it does "boosted MMI" (BMMI).
# On the iterations of training it alternates feature-space
# and model-space training.  We do 8 iterations in total--
# 4 of each type ((B)MMI, f(B)MMI)


# Begin configuration section.
cmd=run.pl
schedule="fmmi fmmi fmmi fmmi mmi mmi mmi mmi"
boost=0.0
learning_rate=0.01
tau=400 # For model.  Note: we're doing smoothing "to the previous iteration",
    # so --smooth-from-model so 400 seems like a more sensible default
    # than 100.  We smooth to the previous iteration because now
    # we are discriminatively training the features (and not using
    # the indirect differential), so it seems like it wouldn't make 
    # sense to use any element of ML.
weight_tau=10 # for model weights.
cancel=true # if true, cancel num and den counts as described in 
     # the boosted MMI paper. 
drop_frames=false # if true, ignore stats from frames where num + den
                       # have no overlap. 
indirect=true # if true, use indirect derivative.
acwt=0.1
stage=-1
ngselect=2; # Just the 2 top Gaussians.  Beyond that, adding more Gaussians
            # wouldn't make much difference since the posteriors would be very small.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: steps/train_mmi_fmmi.sh <data> <lang> <ali-dir> <diag-ubm-dir> <denlat-dir> <exp-dir>"
  echo " e.g.: steps/train_mmi_fmmi.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm2d exp/tri2b_denlats_si84 exp/tri2b_fmmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1) ... boosted MMI."
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  echo "  --learning-rate                                  # learning rate for fMMI, default 0.01"
  echo "  --schedule                                       # learning schedule: by default,"
  echo "                                                   # \"fmmi mmi fmmi mmi fmmi mmi fmmi mmi\""
  exit 1;
fi


data=$1
lang=$2
alidir=$3
dubmdir=$4  # where diagonal UBM is.
denlatdir=$5
dir=$6

silphonelist=`cat $lang/phones/silence.csl`
mkdir -p $dir/log

for f in $data/feats.scp $lang/phones.txt $dubmdir/final.dubm $alidir/final.mdl \
    $alidir/ali.1.gz $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "Expected file $f to exist" && exit 1;
done
cp $alidir/final.mdl $alidir/tree $dir || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;
sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

# Note: $feats is the features before fMPE.
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f $alidir/trans.1 ] && echo Using transforms from $alidir && \
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$alidir/trans.JOB ark:- ark:- |"

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi


fmpefeats="$feats" # At first, the features "after fMPE" are the same as the 
                   # base features.


# Initialize the fMPE object.  Note: we call it .fmpe because
# that's what it was called in the original paper, but since
# we're using the MMI objective function, it's really fMMI.

fmpe-init $dubmdir/final.dubm $dir/0.fmpe 2>$dir/log/fmpe_init.log || exit 1;


if [ $stage -le -1 ]; then
  # Get the gselect (Gaussian selection) info for fMPE.
  # Note: fMPE object starts with GMM object, so can be read
  # as one.
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$ngselect $dir/0.fmpe "$feats" \
    "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

cp $alidir/final.mdl $dir/0.mdl

x=0
num_iters=`echo $schedule | wc -w`

while [ $x -lt $num_iters ]; do
  iter_type=`echo $schedule | cut -d ' ' -f $[$x+1]`
  case $iter_type in 
    fmmi)
    echo "Iteration $x: doing fMMI"
    if [ $stage -le $x ]; then
      numpost="ark,s,cs:gunzip -c $alidir/ali.JOB.gz| ali-to-post ark:- ark:-|"
        # Note: the command gmm-fmpe-acc-stats below requires the pre-fMPE features.
      $cmd JOB=1:$nj $dir/log/acc_fmmi.$x.JOB.log \
        gmm-rescore-lattice $dir/$x.mdl "$lats" "$fmpefeats" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
        sum-post --drop-frames=$drop_frames --scale1=-1 ark:- "$numpost" ark:- \| \
        gmm-fmpe-acc-stats $dir/$x.mdl $dir/$x.fmpe "$feats" \
        "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" ark,s,cs:- \
        $dir/$x.JOB.fmpe_acc || exit 1;
      
      ( fmpe-sum-accs $dir/$x.fmpe_acc $dir/$x.*.fmpe_acc && \
        rm $dir/$x.*.fmpe_acc && \
        fmpe-est --learning-rate=$learning_rate $dir/$x.fmpe $dir/$x.fmpe_acc $dir/$[$x+1].fmpe ) \
        2>$dir/log/est_fmpe.$x.log || exit 1;
    fi
    # We need to set the features to use the correct fMPE object.
    fmpefeats="$feats fmpe-apply-transform $dir/$[$x+1].fmpe ark:- 'ark,s,cs:gunzip -c $dir/gselect.JOB.gz|' ark:- |" 
    rm $dir/$[x+1].mdl 2>/dev/null; ln -s $x.mdl $dir/$[$x+1].mdl # link previous model.
    # Now, diagnostics.
    objf_nf=`grep Overall $dir/log/acc_fmmi.$x.*.log | grep gmm-fmpe-acc-stats | awk '{ p+=$10*$12; nf+=$12; } END{print p/nf, nf;}'`
    objf=`echo $objf_nf | awk '{print $1}'`;
    nf=`echo $objf_nf | awk '{print $2}'`;
    impr=`grep Objf $dir/log/est_fmpe.$x.log | awk '{print $NF}'`
    impr=`perl -e "print ($impr/$nf);"` # normalize by #frames.
    echo On iter $x, objf was $objf, auxf improvement from fMMI was $impr | tee $dir/objf.$x.log
    ;;
    mmi) # MMI iteration.
    echo "Iteration $x: doing MMI (getting stats)..."
    # Get denominator stats...  For simplicity we rescore the lattice
    # on all iterations, even though it shouldn't be necessary on the zeroth
    # (but we want this script to work even if $alidir doesn't contain the
    # model used to generate the lattice).
    if [ $stage -le $x ]; then
      $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
        gmm-rescore-lattice $dir/$x.mdl "$lats" "$fmpefeats" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
        sum-post --drop-frames=$drop_frames --merge=$cancel --scale1=-1 \
        ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
        gmm-acc-stats2 $dir/$x.mdl "$fmpefeats" ark,s,cs:- \
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

      # note: this tau value is for smoothing to model parameters;
      # you need to use gmm-ismooth-stats to smooth to the ML stats,
      # but anyway this script does canceling of num and den stats on
      # each frame (as suggested in the Boosted MMI paper) which would
      # make smoothing to ML impossible without accumulating extra stats.
      $cmd $dir/log/update.$x.log \
        gmm-est-gaussians-ebw --tau=$tau $dir/$x.mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc - \| \
        gmm-est-weights-ebw --weight-tau=$weight_tau - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
    else 
      echo "not doing this iteration because --stage=$stage"
    fi
  
    # Some diagnostics.. note, this objf is somewhat comparable to the
    # MMI objective function divided by the acoustic weight, and differences in it
    # are comparable to the auxf improvement printed by the update program.
    objf_nf=`grep Overall $dir/log/acc.$x.*.log | grep gmm-acc-stats2 | awk '{ p+=$10*$12; nf+=$12; } END{print p/nf, nf;}'`
    objf=`echo $objf_nf | awk '{print $1}'`;
    nf=`echo $objf_nf | awk '{print $2}'`;
    impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
    impr=`perl -e "print ($impr/$nf);"` # renormalize by "real" #frames, to correct
    # for the canceling of stats.
    echo On iter $x, objf was $objf, auxf improvement was $impr | tee $dir/objf.$x.log
    rm $dir/$[x+1].fmpe 2>/dev/null; ln -s $x.fmpe $dir/$[$x+1].fmpe # link previous fMPE transform
    ;;
    *) echo "Invalid --schedule option: expected only mmi or fmmi.";
  esac
  x=$[$x+1]
done

echo "Succeeded with $num_iters iters iterations of MMI+fMMI training (boosting factor = $boost)"

rm $dir/final.mdl 2>/dev/null; ln -s $num_iters.mdl $dir/final.mdl
rm $dir/final.fmpe 2>/dev/null; ln -s $num_iters.fmpe $dir/final.fmpe 

# Now do some cleanup.
rm $dir/gselect.*.gz $dir/*.acc $dir/*.fmpe_acc
exit 0;

