#!/bin/bash


N=10
inv_acwt=12
cmd=scripts/run.pl
use_phi=false  # This is kind of an obscure option.  If true, we'll remove the old
  # LM weights (times 1-RNN_scale) using a phi (failure) matcher, which is
  # appropriate if the old LM weights were added in this way, e.g. by
  # lmrescore.sh.  Otherwise we'll use normal composition, which is appropriate
  # if the lattices came directly from decoding.  This won't actually make much
  # difference (if any) to WER, it's more so we know we are doing the right thing.
for x in `seq 4`; do
  if [ "$1" == "--cmd" ]; then
    cmd=$2
    shift 2
    [ -z "$cmd" ] && echo "Empty argument to --cmd option" && exit 1;
  fi  
  if [ "$1" == "--n" ]; then
    N=$2
    shift 2;
  fi
  if [ "$1" == "--phi" ]; then
    use_phi=true
    shift;
  fi
  if [ "$1" == "--inv-acwt" ]; then # Matters because we use this acwt to 
    # compute the n-best. [later we'll search across different acwt's, 
    # but it'll be limited to the n-best.]
    inv_acwt=$2
    shift 2
  fi
done

acwt=`perl -e "print (1.0/$inv_acwt);"`

if [ $# != 6 ]; then
   echo "Do language model rescoring of lattices (partially remove old LM, add new LM)"
   echo "This version applies an RNNLM and mixes it with the LM scores"
   echo "previously in the lattices., controlled by the first parameter (rnnlm-weight)"
   echo "Note: RNNLM is order-dependent.  The history from the previous sentence is used."
   echo "Usage: scripts/lmrescore.sh <rnn-weight> <old-lang-dir> <rnn-dir> <data-dir> <input-decode-dir> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;


rnnweight=$1
oldlang=$2
rnndir=$3
data=$4
indir=$5
dir=$6


oldlm=$oldlang/G.fst
[ ! -f $oldlm ] && echo Missing file $oldlm && exit 1;
! ls $indir/lat.*.gz >/dev/null && echo "No lattices input directory $indir" && exit 1;



mkdir -p $dir;
phi=`grep -w '#0' $oldlang/words.txt | awk '{print $2}'`

rm $dir/.error 2>/dev/null
mkdir -p $dir/log

for lat in $indir/lat.*.gz; do
  n=`basename $lat | cut -d. -f2`;
  newlat=$dir/`basename $lat`

  # First convert lattice to N-best.  Be careful because this
  # will be quite sensitive to the acoustic scale; this should be close
  # to the one we'll finally get the best WERs with.
  # Note: the lattice-rmali part here is just because we don't
  # need the alignments for what we're doing.
 
  ( lattice-to-nbest --acoustic-scale=$acwt --n=$N \
    "ark:gunzip -c $lat|" ark:- | \
    lattice-rmali ark:- "ark:|gzip -c >$dir/nbest1.$n.gz" ) 2>$dir/log/lat2nbest.$n.log || exit 1;

  # next remove part of the old LM probs.  
  if $use_phi; then # we'll use the phi-matcher style of composition.. this is appropriate
                # if the old LM scores were added e.g. by lmrescore.sh, using this style.
    ( gunzip -c $dir/nbest1.$n.gz | \
      lattice-compose --phi-label=$phi ark:- $oldlm ark:- | \
      gzip -c >$dir/nbest2.$n.gz ) 2>$dir/log/remove_old.$n.log || exit 1;

  else # this approach chooses the best path through the old LM FST.
    ( gunzip -c $dir/nbest1.$n.gz | \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- | \
      lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- | \
      lattice-1best ark:- ark:- | \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- | \
      gzip -c >$dir/nbest2.$n.gz ) 2>$dir/log/remove_old.$n.log || exit 1;
  fi
  # Next decompose the n-best lists into 4 archives.
  mkdir -p $dir/archives.$n
  adir=$dir/archives.$n
  nbest-to-linear "ark:gunzip -c $dir/nbest2.$n.gz|" \
    "ark,t:$adir/ali" "ark,t:$adir/words" \
    "ark,t:$adir/lmwt.nolm" "ark,t:$adir/acwt" 2>$dir/log/to_linear1.$n.log || exit 1;
  # We also want an archive with the LM scores before we
  # removed the LM probs (will help us do interpolation).
  nbest-to-linear "ark:gunzip -c $dir/nbest1.$n.gz|" "ark:/dev/null" \
    "ark:/dev/null" "ark,t:$adir/lmwt.withlm" "ark:/dev/null" \
    2>$dir/log/to_linear2.$n.log || exit 1;

 # Below was debug to make sure we could get the original results
 # (at and close to the acwt where we generated the N-best).
 # # At this point, let's just reconstruct into N-best lists with
 # # the old LM scores present, and verify that we get the same output.
 # # (well, modulo the inexactness of doing N-best; it should only be
 # # the exact same output at the acwt we used to make the N-best.
 # ( linear-to-nbest "ark:$adir/ali" "ark:$adir/words" "ark:$adir/lmwt.withlm" \
 #   "ark:$adir/acwt" ark:- | \
 #   nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.$n.gz" ) \
 #    2>$dir/log/linear_to_lattice.$n.log || exit 1;

 scripts/int2sym.pl --ignore-first-field $oldlang/words.txt <$adir/words >$adir/words_text

 mkdir -p $adir/temp
 paste $adir/lmwt.nolm $adir/lmwt.withlm | awk '{print $1, ($4-$2);}' > \
    $adir/lmwt.lmonly

 scripts/rnnlm_compute_scores.sh $rnndir $adir/temp $adir/words_text $adir/lmwt.rnn
 paste $adir/lmwt.nolm $adir/lmwt.lmonly $adir/lmwt.rnn | awk -v rnnweight=$rnnweight \
   '{ key=$1; graphscore=$2; lmscore=$4; rnnscore=$6; 
     score = graphscore+(rnnweight*rnnscore)+((1-rnnweight)*lmscore);
     print $1,score; } ' > $adir/lmwt.interp.$rnnscore

 # Convert back into lattices, with the new LM score.
 ( linear-to-nbest "ark:$adir/ali" "ark:$adir/words" "ark:$adir/lmwt.interp.$rnnscore" \
   "ark:$adir/acwt" ark:- | \
   nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.$n.gz" ) \
    2>$dir/log/linear_to_lattice.$n.log || exit 1;

done


#wait
#[ -f $dir/.error ] && echo Error doing LM rescoring && exit 1

scripts/score_lats.sh $dir $oldlang/words.txt $data

