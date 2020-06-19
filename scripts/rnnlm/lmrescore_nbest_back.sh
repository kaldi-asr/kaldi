#!/usr/bin/env bash

# Copyright 2017  Hainan Xu
#           2017  Szu-Jui Chen

# This script is very similar to scripts/rnnlm/lmrescore_nbest.sh, and it takes the results
# from forward model then performs n-best LM rescoring based on backward model with Kaldi-RNNLM.

# Begin configuration section.
N=10
inv_acwt=10
cmd=run.pl
use_phi=false  # This is kind of an obscure option.  If true, we'll remove the old
  # LM weights (times 1-RNN_scale) using a phi (failure) matcher, which is
  # appropriate if the old LM weights were added in this way, e.g. by
  # lmrescore.sh.  Otherwise we'll use normal composition, which is appropriate
  # if the lattices came directly from decoding.  This won't actually make much
  # difference (if any) to WER, it's more so we know we are doing the right thing.
test=false # Activate a testing option.
stage=1 # Stage of this script, for partial reruns.
skip_scoring=false
keep_ali=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# != 6 ]; then
   echo "Do language model rescoring of lattices (partially remove old LM, add new LM)"
   echo "This version applies an RNNLM and mixes it with the LM scores"
   echo "previously in the lattices., controlled by the first parameter (rnnlm-weight)"
   echo ""
   echo "Usage: $0 [options] <rnn-weight> <old-lang-dir> <rnn-dir> <data-dir> <input-decode-dir> <output-decode-dir>"
   echo "Main options:"
   echo "  --inv-acwt <inv-acwt>          # default 12.  e.g. --inv-acwt 17.  Equivalent to LM scale to use."
   echo "                                 # for N-best list generation... note, we'll score at different acwt's"
   echo "  --cmd <run.pl|queue.pl [opts]> # how to run jobs."
   echo "  --phi (true|false)             # Should be set to true if the source lattices were created"
   echo "                                 # by lmrescore.sh, false if they came from decoding."
   echo "  --N <N>                        # Value of N in N-best rescoring (default: 10)"
   exit 1;
fi

rnnweight=$1
oldlang=$2
rnndir=$3
data=$4
indir=$5
dir=$6

acwt=`perl -e "print (1.0/$inv_acwt);"`

# Figures out if the old LM is G.fst or G.carpa
oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

for f in $rnndir/final.raw $data/feats.scp $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

nj=`cat $indir/num_jobs` || exit 1;
mkdir -p $dir;
cp $indir/num_jobs $dir/num_jobs

adir=$dir/archives

phi=`grep -w '#0' $oldlang/words.txt | awk '{print $2}'`

rm $dir/.error 2>/dev/null
mkdir -p $dir/log

# First convert lattice to N-best.  Be careful because this
# will be quite sensitive to the acoustic scale; this should be close
# to the one we'll finally get the best WERs with.
# Note: the lattice-rmali part here is just because we don't
# need the alignments for what we're doing.
if [ $stage -le 5 ]; then
  echo "$0: Copying needed information from $indir/archives to $adir"
    # Do some small tasks; for these we don't use the queue, it will only slow us down.
  for n in `seq $nj`; do
    mkdir -p $adir.$n
    cp $indir/archives.$n/ali $adir.$n/
    cp $indir/archives.$n/words $adir.$n/
    cp $indir/archives.$n/words_text $adir.$n/
    cp $indir/archives.$n/lmwt.nolm $adir.$n/
    cp $indir/archives.$n/acwt $adir.$n/
    cp $indir/archives.$n/lmwt.withlm $adir.$n/
    
    mkdir -p $adir.$n/temp
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.withlm | awk '{print $1, ($4-$2);}' > \
      $adir.$n/lmwt.lmonly || exit 1;
  done
fi
if [ $stage -le 6 ]; then
  echo "$0: invoking rnnlm/compute_sentence_scores_back.sh which calls rnnlm to get RNN LM scores."
  $cmd JOB=1:$nj $dir/log/rnnlm_compute_scores.JOB.log \
    rnnlm/compute_sentence_scores_back.sh $rnndir $adir.JOB/temp \
                                   $adir.JOB/words_text $adir.JOB/lmwt.rnn 
fi

if [ $stage -le 7 ]; then
  echo "$0: doing average on forward and backward scores."
  for n in `seq $nj`; do
    paste $indir/archives.$n/lmwt.rnn $adir.$n/lmwt.rnn | awk -F' ' '{print $1,$2 * 0.5 + $4 * 0.5}' \
    > $adir.$n/lmwt.rnn_bi
  done
fi

if [ $stage -le 8 ]; then
  echo "$0: reconstructing total LM+graph scores including interpolation of RNNLM and old LM scores."
  for n in `seq $nj`; do
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.lmonly $adir.$n/lmwt.rnn_bi | awk -v rnnweight=$rnnweight \
      '{ key=$1; graphscore=$2; lmscore=$4; rnnscore=$6;
     score = graphscore+(rnnweight*rnnscore)+((1-rnnweight)*lmscore);
     print $1,score; } ' > $adir.$n/lmwt.interp.$rnnweight || exit 1;
  done
fi

if [ $stage -le 9 ]; then
  echo "$0: reconstructing archives back into lattices."
  $cmd JOB=1:$nj $dir/log/reconstruct_lattice.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" \
    "ark:$adir.JOB/lmwt.interp.$rnnweight" "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;

