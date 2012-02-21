#!/bin/bash
# by Dan Povey, 2012.  Apache.

# This script does MMI discriminative training, including
# feature-space (like fMPE) and model-space components. 
# If you give the --boost option it does "boosted MMI" (BMMI).
# On the iterations of training it alternates feature-space
# and model-space training.  We do 8 iterations in total--
# 4 of each type ((B)MMI, f(B)MMI)

# The features it uses are LDA + [something], where the something
# may be just a global transform like MLLT, or may also include
# speaker-specific transforms such as SAT.  This script just uses
# transforms computed in the alignment directory, so it doesn't
# need to know what the transform type is (it isn't re-estimating
# them itself)


niters=8
nj=4
boost=0.0
lrate=0.01
tau=200 # Note: we're doing smoothing "to the previous iteration"
    # --smooth-from-model so 200 seems like a more sensible default
    # than 100.  We smooth to the previous iteration because now
    # we are discriminatively training the features (and not using
    # the indirect differential), so it seems like it wouldn't make 
    # sense to use any element of ML.
ngauss=400
merge=true # if true, cancel num and den counts as described in 
    # the boosted MMI paper. 


cmd=scripts/run.pl
acwt=0.1
stage=-1

for x in `seq 8`; do
  if [ $1 == "--num-jobs" ]; then
    shift; nj=$1; shift
  fi
  if [ $1 == "--learning-rate" ]; then
    shift; lrate=$1; shift
  fi
  if [ $1 == "--num-gauss" ]; then
    shift; ngauss=$1; shift  # #Gauss in GMM for fMPE.
  fi
  if [ $1 == "--num-iters" ]; then
    shift; niters=$1; shift
  fi
  if [ $1 == "--boost" ]; then
    shift; boost=$1; shift
  fi
  if [ $1 == "--cmd" ]; then
    shift; cmd=$1; shift
    [ -z "$cmd" ] && echo Empty argument to --cmd option && exit 1;
  fi  
  if [ $1 == "--acwt" ]; then
    shift; acwt=$1; shift
  fi  
  if [ $1 == "--tau" ]; then
    shift; tau=$1; shift
  fi  
  if [ $1 == "--stage" ]; then # used for finishing partial runs.
    shift; stage=$1; shift
  fi  
done

if [ $# != 7 ]; then
   echo "Usage: steps/train_lda_etc_mmi_fmmi.sh <data-dir> <lang-dir> <ali-dir> <dubm-dir> <denlat-dir> <model-dir> <exp-dir>"
   echo " e.g.: steps/train_lda_etc_mmi_fmmi.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm2d exp/tri2b_denlats_si84 exp/tri2b exp/tri2b_fmmi"
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

data=$1
lang=$2
alidir=$3
dubmdir=$4  # where diagonal UBM is.
denlatdir=$5
srcdir=$6 # may be same model as in alidir, but may not be, e.g.
      # if you want to test MMI with different #iters.
dir=$7
silphonelist=`cat $lang/silphones.csl`
ngselect=2; # Just the 2 top Gaussians.  Beyond that wouldn't make much
   # difference since the posteriors would be very small.
mkdir -p $dir/log

if [ ! -f $srcdir/final.mdl -o ! -f $srcdir/final.mat ]; then
  echo "Error: alignment dir $alidir does not contain one of final.mdl or final.mat"
  exit 1;
fi
cp $srcdir/final.mat $srcdir/tree $dir

n=`get_splits.pl $nj | awk '{print $1}'`
if [ -f $alidir/$n.trans ]; then
  use_trans=true
  echo Using transforms from directory $alidir
else
  echo No transforms present in alignment directory: assuming speaker independent.
  use_trans=false
fi

# Note: ${basefeatspart[$n]} is the features before fMPE.

for n in `get_splits.pl $nj`; do
  basefeatspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  $use_trans && basefeatspart[$n]="${basefeatspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  featspart[$n]="${basefeatspart[$n]}" # before 1st iter of fMPE..

  [ ! -f $denlatdir/lat.$n.gz ] && echo No such file $denlatdir/lat.$n.gz && exit 1;
  latspart[$n]="ark:gunzip -c $denlatdir/lat.$n.gz|"
  # note: in next line, doesn't matter which model we use, it's only used to map to phones.
  [ $boost != "0.0" -a $boost != "0" ] && latspart[$n]="${latspart[$n]} lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/$n.ali.gz|' ark:- |"
done


# Initialize the fMPE object.  Note: we call it .fmpe because
# that's what it was called in the original paper, but since
# we're using the MMI objective function, it's really fMMI.
fmpe-init $dubmdir/final.dubm $dir/0.fmpe || exit 1;

rm $dir/.error 2>/dev/null

if [ $stage -le -1 ]; then
# Get the gselect (Gaussian selection) info for fMPE.
# Note: fMPE object starts with GMM object, so can be read
# as one.
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/gselect.$n.log \
      gmm-gselect --n=$ngselect $dir/0.fmpe "${featspart[$n]}" \
      "ark:|gzip -c >$dir/gselect.$n.gz" || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error in Gaussian selection phase" && exit 1;
fi


cur_mdl=$srcdir/final.mdl
cur_fmpe=$dir/0.fmpe
x=0
while [ $x -lt $niters ]; do
  if [ $[$x%2] == 0 ]; then
    echo "Iteration $x: doing fMMI"
    if [ $stage -le $x ]; then
      for n in `get_splits.pl $nj`; do  
        numpost="ark,s,cs:gunzip -c $alidir/$n.ali.gz| ali-to-post ark:- ark:-|"
        # Note: the command gmm-fmpe-acc-stats below requires the "base" features
        # (without fMPE), not the fMPE features.
        $cmd $dir/log/acc_fmmi.$x.$n.log \
         gmm-rescore-lattice $cur_mdl "${latspart[$n]}" "${featspart[$n]}" ark:- \| \
          lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
          sum-post --scale1=-1 ark:- "$numpost" ark:- \| \
          gmm-fmpe-acc-stats $cur_mdl $cur_fmpe "${basefeatspart[$n]}" \
           "ark,s,cs:gunzip -c $dir/gselect.$n.gz|" ark,s,cs:- \
           $dir/$x.$n.fmpe_acc || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && echo Error doing fMPE accumulation && exit 1;
      ( sum-matrices $dir/$x.fmpe_acc $dir/$x.*.fmpe_acc && \
        rm $dir/$x.*.fmpe_acc && \
        fmpe-est --learning-rate=$lrate $cur_fmpe $dir/$x.fmpe_acc $dir/$[$x+1].fmpe ) \
       2>$dir/log/est_fmpe.$x.log || exit 1;
      rm $dir/$[$x+1].mdl 2>/dev/null
    fi
    # We need to set the features to use the correct fMPE object.
    for n in `get_splits.pl $nj`; do
      featspart[$n]="${basefeatspart[$n]} fmpe-apply-transform $dir/$[$x+1].fmpe ark:- 'ark,s,cs:gunzip -c $dir/gselect.$n.gz|' ark:- |" 
    done      
    cur_fmpe=$dir/$[$x+1].fmpe
    # Now, diagnostics.
    objf=`grep Overall $dir/log/acc_fmmi.$x.*.log | grep gmm-fmpe-acc-stats | awk '{ p+=$10*$12; nf+=$12; } END{print p/nf;}'`
    nf=`grep Overall $dir/log/acc_fmmi.$x.*.log | grep gmm-fmpe-acc-stats | awk '{ nf+=$12; } END{print nf;}'`
    impr=`grep Objf $dir/log/est_fmpe.$x.log | awk '{print $NF}'`
    impr=`perl -e "print ($impr/$nf);"` # normalize by #frames.
    echo On iter $x, objf was $objf, auxf improvement from fMMI was $impr | tee $dir/objf.$x.log
  else
    echo "Iteration $x: doing MMI (getting stats)..."
    # Get denominator stats...  For simplicity we rescore the lattice
    # on all iterations, even though it shouldn't be necessary on the zeroth
    # (but we want this script to work even if $srcdir doesn't contain the
    # model used to generate the lattice).
    if [ $stage -le $x ]; then
      for n in `get_splits.pl $nj`; do  
        $cmd $dir/log/acc.$x.$n.log \
          gmm-rescore-lattice $cur_mdl "${latspart[$n]}" "${featspart[$n]}" ark:- \| \
          lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
          sum-post --merge=$merge --scale1=-1 \
          ark:- "ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- |" ark:- \| \
          gmm-acc-stats2 $cur_mdl "${featspart[$n]}" ark,s,cs:- \
          $dir/num_acc.$x.$n.acc $dir/den_acc.$x.$n.acc  || touch $dir/.error &
      done 
      wait
      [ -f $dir/.error ] && echo Error accumulating stats on iter $x && exit 1;
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
        gmm-est-gaussians-ebw --tau=$tau $cur_mdl $dir/num_acc.$x.acc $dir/den_acc.$x.acc - \| \
        gmm-est-weights-ebw - $dir/num_acc.$x.acc $dir/den_acc.$x.acc $dir/$[$x+1].mdl || exit 1;
    else 
      echo "not doing this iteration because --stage=$stage"
    fi
  
    # Some diagnostics.. note, this objf is somewhat comparable to the
    # MMI objective function divided by the acoustic weight, and differences in it
    # are comparable to the auxf improvement printed by the update program.
    objf=`grep Overall $dir/log/acc.$x.*.log | grep gmm-acc-stats2 | awk '{ p+=$10*$12; nf+=$12; } END{print p/nf;}'`
    nf=`grep Overall $dir/log/acc.$x.*.log | grep gmm-acc-stats2 | awk '{ nf+=$12; } END{print nf;}'`
    impr=`grep Overall $dir/log/update.$x.log | head -1 | awk '{print $10*$12;}'`
    impr=`perl -e "print ($impr/$nf);"` # renormalize by "real" #frames, to correct
    # for the canceling of stats.
    echo On iter $x, objf was $objf, auxf improvement was $impr | tee $dir/objf.$x.log
    cur_mdl=$dir/$[$x+1].mdl
  fi
  x=$[$x+1]
done

echo "Succeeded with $niters iterations of MMI+fMMI training (boosting factor = $boost)"

( cd $dir; rm final.mdl 2>/dev/null; ln -s `basename $cur_mdl` final.mdl;
  rm final.fmpe 2>/dev/null; ln -s `basename $cur_fmpe` final.fmpe )

# Now do some cleanup.
rm $dir/gselect.*.gz $dir/*.acc $dir/*.fmpe_acc
