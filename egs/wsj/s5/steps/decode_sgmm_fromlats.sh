#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does decoding with an SGMM system, with speaker vectors. 
# If the SGMM system was
# built on top of fMLLR transforms from a conventional system, you should
# provide the --transform-dir option.
# This script does not use a decoding graph, but instead you provide
# a previous decoding directory with lattices in it.  This script will only
# make use of the word sequences in the lattices; it limits the decoding
# to those sequences.  You should also provide a "lang" directory from 
# which this script will use the G.fst and L.fst.

# Begin configuration section.
stage=1
alignment_model=
transform_dir=    # dir to find fMLLR transforms.
acwt=0.08333  # Just a default value, used for adaptation and beam-pruning..
batch_size=75 # Limits memory blowup in compile-train-graphs-fsts
cmd=run.pl
beam=20.0
gselect=15  # Number of Gaussian-selection indices for SGMMs.  [Note:
            # the first_pass_gselect variable is used for the 1st pass of
            # decoding and can be tighter.
first_pass_gselect=3 # Use a smaller number of Gaussian-selection indices in 
            # the 1st pass of decoding (lattice generation).
max_active=7000

#WARNING: This option is renamed lattice_beam (it was renamed to follow the naming 
#         in the other scripts
lattice_beam=8.0 # Beam we use in lattice generation.
vecs_beam=4.0 # Beam we use to prune lattices while getting posteriors for 
    # speaker-vector computation.  Can be quite tight (actually we could
    # probably just do best-path.
use_fmllr=false
fmllr_iters=10
fmllr_min_count=1000
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"

# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: steps/decode_sgmm_fromlats.sh [options] <data-dir> <lang-dir> <old-decode-dir> <decode-dir>"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --alignment-model <ali-mdl>              # Model for the first-pass decoding."
  echo "  --config <config-file>                   # config containing options"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 13.0"
  exit 1;
fi

data=$1
lang=$2
olddir=$3
dir=$4
srcdir=`dirname $dir`

for f in $data/feats.scp $lang/G.fst $lang/L_disambig.fst $lang/phones/disambig.int \
    $srcdir/final.mdl $srcdir/tree $olddir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $olddir/num_jobs` || exit 1;
sdata=$data/split$nj;
silphonelist=`cat $lang/phones/silence.csl` || exit 1
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"
gselect_opt_1stpass="$gselect_opt copy-gselect --n=$first_pass_gselect ark:- ark:- |"

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"
if [ -z "$transform_dir" ] && [ -f $olddir/trans.1 ]; then
  transform_dir=$olddir
fi

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi

## Calculate FMLLR pre-transforms if needed. We are doing this here since this
## step is requried by models both with and without speaker vectors
if $use_fmllr; then
  if [ ! -f $srcdir/final.fmllr_mdl ] || [ $srcdir/final.fmllr_mdl -ot $srcdir/final.mdl ]; then
    echo "$0: computing pre-transform for fMLLR computation."
    sgmm-comp-prexform $srcdir/final.mdl $srcdir/final.occs $srcdir/final.fmllr_mdl || exit 1;
  fi
fi

## Save Gaussian-selection info to disk.
# Note: we can use final.mdl regardless of whether there is an alignment model--
# they use the same UBM.
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    sgmm-gselect --full-gmm-nbest=$gselect $srcdir/final.mdl \
    "$feats" "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

## Work out name of alignment model. ##
if [ -z "$alignment_model" ]; then
  if [ -f "$srcdir/final.alimdl" ]; then alignment_model=$srcdir/final.alimdl;
  else alignment_model=$srcdir/final.mdl; fi
fi
[ ! -f "$alignment_model" ] && echo "$0: no alignment model $alignment_model " && exit 1;

# Generate state-level lattice which we can rescore.  This is done with the 
# alignment model and no speaker-vectors.
if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $dir/log/decode_pass1.JOB.log \
 lattice-to-fst "ark:gunzip -c $olddir/lat.JOB.gz|" ark:- \| \
  fsttablecompose "fstproject --project_output=true $lang/G.fst | fstarcsort |" ark:- ark:- \| \
  fstdeterminizestar ark:- ark:- \| \
  compile-train-graphs-fsts --read-disambig-syms=$lang/phones/disambig.int \
    --batch-size=$batch_size $scale_opts \
    $srcdir/tree $srcdir/final.mdl $lang/L_disambig.fst ark:- ark:- \| \
  sgmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --determinize-lattice=false --allow-partial=true \
    --word-symbol-table=$lang/words.txt "$gselect_opt_1stpass" $alignment_model \
    "ark:-" "$feats" "ark:|gzip -c > $dir/pre_lat.JOB.gz" || exit 1;
fi

## Check if the model has speaker vectors
spkdim=`sgmm-info $srcdir/final.mdl | grep 'speaker vector' | awk '{print $NF}'`

if [ $spkdim -gt 0 ]; then  ### For models with speaker vectors:

# Estimate speaker vectors (1st pass).  Prune before determinizing
# because determinization can take a while on un-pruned lattices.
# Note: the sgmm-post-to-gpost stage is necessary because we have
# a separate alignment-model and final model, otherwise we'd skip it 
# and use sgmm-est-spkvecs.
  if [ $stage -le 3 ]; then
    $cmd JOB=1:$nj $dir/log/vecs_pass1.JOB.log \
      gunzip -c $dir/pre_lat.JOB.gz \| \
      lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      weight-silence-post 0.0 $silphonelist $alignment_model ark:- ark:- \| \
      sgmm-post-to-gpost "$gselect_opt" $alignment_model "$feats" ark:- ark:- \| \
      sgmm-est-spkvecs-gpost --spk2utt=ark:$sdata/JOB/spk2utt \
      $srcdir/final.mdl "$feats" ark,s,cs:- "ark:$dir/pre_vecs.JOB" || exit 1;
  fi

# Estimate speaker vectors (2nd pass).  Since we already have spk vectors,
# at this point we need to rescore the lattice to get the correct posteriors.
  if [ $stage -le 4 ]; then
    $cmd JOB=1:$nj $dir/log/vecs_pass2.JOB.log \
      gunzip -c $dir/pre_lat.JOB.gz \| \
      sgmm-rescore-lattice --spk-vecs=ark:$dir/pre_vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk \
      "$gselect_opt" $srcdir/final.mdl ark:- "$feats" ark:- \| \
      lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- \| \
      sgmm-est-spkvecs --spk2utt=ark:$sdata/JOB/spk2utt "$gselect_opt" --spk-vecs=ark:$dir/pre_vecs.JOB \
      $srcdir/final.mdl "$feats" ark,s,cs:- "ark:$dir/vecs.JOB" || exit 1;
  fi
  rm $dir/pre_vecs.*

  if $use_fmllr; then
  # Estimate fMLLR transforms (note: these may be on top of any
  # fMLLR transforms estimated with the baseline GMM system.
    if [ $stage -le 5 ]; then # compute fMLLR transforms.
      echo "$0: computing fMLLR transforms."
      $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
	gunzip -c $dir/pre_lat.JOB.gz \| \
	sgmm-rescore-lattice --spk-vecs=ark:$dir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk \
	"$gselect_opt" $srcdir/final.mdl ark:- "$feats" ark:- \| \
	lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
	lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
	lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
	weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- \| \
	sgmm-est-fmllr --spk2utt=ark:$sdata/JOB/spk2utt "$gselect_opt" --spk-vecs=ark:$dir/vecs.JOB \
	--fmllr-iters=$fmllr_iters --fmllr-min-count=$fmllr_min_count \
	$srcdir/final.fmllr_mdl "$feats" ark,s,cs:- "ark:$dir/trans.JOB" || exit 1;
    fi
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"  
  fi

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.
  if [ $stage -le 6 ]; then
    $cmd JOB=1:$nj $dir/log/rescore.JOB.log \
      sgmm-rescore-lattice "$gselect_opt" --utt2spk=ark:$sdata/JOB/utt2spk --spk-vecs=ark:$dir/vecs.JOB \
      $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.JOB.gz|" "$feats" ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$lattice_beam ark:- \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
  fi
  rm $dir/pre_lat.*.gz

else  ### For models without speaker vectors:

  if $use_fmllr; then
  # Estimate fMLLR transforms (note: these may be on top of any
  # fMLLR transforms estimated with the baseline GMM system.
    if [ $stage -le 5 ]; then # compute fMLLR transforms.
      echo "$0: computing fMLLR transforms."
      $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
	gunzip -c $dir/pre_lat.JOB.gz \| \
	sgmm-rescore-lattice --utt2spk=ark:$sdata/JOB/utt2spk \
	"$gselect_opt" $srcdir/final.mdl ark:- "$feats" ark:- \| \
	lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
	lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
	lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
	weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- \| \
	sgmm-est-fmllr --spk2utt=ark:$sdata/JOB/spk2utt "$gselect_opt" \
	--fmllr-iters=$fmllr_iters --fmllr-min-count=$fmllr_min_count \
	$srcdir/final.fmllr_mdl "$feats" ark,s,cs:- "ark:$dir/trans.JOB" || exit 1;
    fi
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"  
  fi

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.
  if [ $stage -le 6 ] && $use_fmllr; then
    $cmd JOB=1:$nj $dir/log/rescore.JOB.log \
      sgmm-rescore-lattice "$gselect_opt" --utt2spk=ark:$sdata/JOB/utt2spk \
      $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.JOB.gz|" "$feats" ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$lattice_beam ark:- \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
    rm $dir/pre_lat.*.gz
  else  # Already done with decoding if no adaptation needed.
    for n in `seq 1 $nj`; do
      mv $dir/pre_lat.${n}.gz $dir/lat.${n}.gz
    done
  fi

fi

# The output of this script is the files "lat.*.gz"-- we'll rescore this at 
# different acoustic scales to get the final output.


if [ $stage -le 7 ]; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  echo "score best paths"
  local/score.sh --cmd "$cmd" $data $lang $dir
  echo "score confidence and timing with sclite"
  #local/score_sclite_conf.sh --cmd "$cmd" --language turkish $data $lang $dir
fi
echo "Decoding done."
exit 0;
