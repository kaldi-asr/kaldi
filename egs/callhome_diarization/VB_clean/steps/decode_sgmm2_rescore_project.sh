#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does decoding with an SGMM system, by rescoring lattices
# generated from a previous SGMM system.  This version does the "predictive"
# SGMM, where we subtract some constant times the log-prob of the left
# few spliced frames, and the same for the right few.
# The directory with the lattices
# is assumed to contain any speaker vectors, if used.  This script just
# adds into the acoustic scores, (some constant, default -0.25) times
# the acoustic score of the left model, and the same for the right model.

# the lattices one final time, using the same setup as the final decoding
# pass of the source dir.  The assumption is that the model may have
# been discriminatively trained.

# If the system was built on top of fMLLR transforms from a conventional system,
# you should provide the --transform-dir option.

# Begin configuration section.
stage=0
transform_dir=    # dir to find fMLLR transforms.
cmd=run.pl
iter=final
prob_scale=-0.25
dimensions=0:13:104:117
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: steps/decode_sgmm_rescore_project.sh [options] <full-lda-mat> <graph-dir|lang-dir> <data-dir> <old-decode-dir> <decode-dir>"
  echo " e.g.: steps/decode_sgmm_rescore_project.sh --transform-dir exp/tri3b/decode_dev93_tgpr \\"
  echo "     exp/tri2b/full.mat exp/sgmm3a/graph_tgpr data/test_dev93 exp/sgmm3a/decode_dev93_tgpr exp/sgmm3a/decode_dev93_tgpr_predict"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --prob-scale <scale>                     # Default -0.25, scale on left and right models."
  exit 1;
fi

full_lda_mat=$1
graphdir=$2
data=$3
olddir=$4
dir=$5
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

for f in $full_lda_mat $graphdir/words.txt $data/feats.scp $olddir/lat.1.gz \
   $olddir/gselect.1.gz $srcdir/$iter.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $olddir/num_jobs` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -f $olddir/vecs.1 ]; then
  echo "$0: using speaker vectors from $olddir"
  spkvecs_opt="--spk-vecs=ark:$olddir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk"
else
  echo "$0: no speaker vectors found."
  spkvecs_opt=
fi

if [ $stage -le 0 ]; then
  # Get full LDA+MLLT mat and its inverse.  Note: the full LDA+MLLT mat is
  # the LDA+MLLT mat, plus the "rejected" rows of the LDA matrix.
  $cmd $dir/log/get_full_lda.log \
    get-full-lda-mat $srcdir/final.mat $full_lda_mat $dir/full.mat $dir/full_inv.mat || exit 1;
fi

if [ $stage -le 1 ]; then
  left_start=`echo $dimensions | cut '-d:' -f 1`;
  left_end=`echo $dimensions | cut '-d:' -f 2`;
  right_start=`echo $dimensions | cut '-d:' -f 3`;
  right_end=`echo $dimensions | cut '-d:' -f 4`;

  # Prepare left and right models.  For now, the dimensions are hardwired (e.g., 13 MFCCs and splice 9 frames).
  # Note: the choice of dividing by the prob of the left 4 and the right 4 frames is a bit arbitrary and
  # we could investigate different configurations.
  $cmd $dir/log/left.log \
    sgmm2-project --start-dim=$left_start --end-dim=$left_end $srcdir/final.mdl $dir/full.mat $dir/left.mdl $dir/left.mat || exit 1;
  $cmd $dir/log/right.log \
    sgmm2-project --start-dim=$right_start --end-dim=$right_end $srcdir/final.mdl $dir/full.mat $dir/right.mdl $dir/right.mat || exit 1;
fi


# we apply the scaling on the new acoustic probs by adding the inverse
# of that to the old acoustic probs, and then later inverting again.
# this has to do with limitations in sgmm2-rescore-lattice: we can only
# scale the *old* acoustic probs, not the new ones.
inverse_prob_scale=`perl -e "print (1.0 / $prob_scale);"`
cur_lats="ark:gunzip -c $olddir/lat.JOB.gz | lattice-scale --acoustic-scale=$inverse_prob_scale ark:- ark:- |"

## Set up features.  Note: we only support LDA+MLLT features, this
## is inherent in the method, we could not support deltas.

for model_type in left right; do

  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |" # spliced features.
  if [ ! -z "$transform_dir" ]; then  # using speaker-specific transforms.
     # we want to transform in the sequence: $dir/full.mat, then the result of
     # (extend-transform-dim $transform_dir/trans.JOB), then $dir/full_inv.mat to
     # get back to the spliced space, then the left.mat or right.mat.  But
     # note that compose-transforms operates in matrix-multiplication order,
     # which is opposite from the "order of applying the transforms" order.
     new_dim=$[`copy-matrix --binary=false $dir/full.mat - | wc -l` - 1]; # 117 in normal case.
     feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk 'ark:extend-transform-dim --new-dimension=$new_dim ark:$transform_dir/trans.JOB ark:- | compose-transforms ark:- $dir/full.mat ark:- | compose-transforms $dir/full_inv.mat ark:- ark:- | compose-transforms $dir/${model_type}.mat ark:- ark:- |' ark:- ark:- |"
  else  # else, we transform with the "left" or "right" matrix; these transform from the
        # spliced space.
     feats="$feats transform-feats $dir/${model_type}.mat |"
     # If we don't have the --transform-dir option, make sure the model was
     # trained in the same way.
     if grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
       echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
       echo "  but you are not providing the --transform-dir option in test time."
     fi
  fi
  if [ -f $olddir/trans.1 ]; then
     echo "$0: warning: not using transforms in $olddir (this is just a "
     echo " limitation of the script right now, and could be fixed)."
  fi
  
  if [ $stage -le 2 ]; then
    echo "Getting gselect info for $model_type model."
    $cmd JOB=1:$nj $dir/log/gselect.$model_type.JOB.log \
       sgmm2-gselect $dir/$model_type.mdl "$feats" \
       "ark,t:|gzip -c >$dir/gselect.$model_type.JOB.gz" || exit 1;
  fi
  gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.$model_type.JOB.gz|"


  # Rescore the state-level lattices with the model provided.  Just
  # one command in this script.
  # The --old-acoustic-scale=1.0 option means we just add the scores
  # to the old scores.
  if [ $stage -le 3 ]; then
    echo "$0: rescoring lattices with $model_type model"
    $cmd JOB=1:$nj $dir/log/rescore.${model_type}.JOB.log \
      sgmm2-rescore-lattice --old-acoustic-scale=1.0 "$gselect_opt" $spkvecs_opt \
      $dir/$model_type.mdl "$cur_lats" "$feats" \
      "ark:|gzip -c > $dir/lat.${model_type}.JOB.gz" || exit 1;
  fi
  cur_lats="ark:gunzip -c $dir/lat.${model_type}.JOB.gz |"
done

if [ $stage -le 4 ]; then
  echo "$0: getting final lattices."
  $cmd JOB=1:$nj $dir/log/scale_lats.JOB.log \
    lattice-scale --acoustic-scale=$prob_scale "$cur_lats" "ark:|gzip -c >$dir/lat.JOB.gz" \
   || exit 1;
fi

rm $dir/lat.{left,right}.*.gz 2>/dev/null  # note: if these still exist, it will
 # confuse the scoring script.

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
