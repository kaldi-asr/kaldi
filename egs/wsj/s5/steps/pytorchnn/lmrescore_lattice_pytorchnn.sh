#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2021  Johns Hopkins University (Author: Ke Li)

# This script performs lattice rescoring with a PyTorch-trained neural LM.

# Begin configuration section.
model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=768
hidden_dim=768
nlayers=6
nhead=8

inv_acwt=10
weight=0.8 # interpolation weight of a neural network LM with a N-gram LM
beam=5
epsilon=0.5
oov_symbol="'<unk>'"

cmd=run.pl
stage=0
skip_scoring=false
# End configuration section.

echo "$0 $*" # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh


if [ $# != 6 ]; then
   echo "Perform lattice rescoring with a PyTorch-trained neural language model."
   echo "The neural LM is interpolated with an N-gram LM during rescoring."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <nn-model-dir> vocab <data-dir> <input-decode-dir> <output-decode-dir>"
   echo "Main options:"
   echo "  --inv-acwt <inv-acwt>          # default 12.  e.g. --inv-acwt 17.  Equivalent to LM scale to use."
   echo "                                 # for N-best list generation... note, we'll score at different acwt's"
   echo "  --cmd <run.pl|queue.pl [opts]> # how to run jobs."
   exit 1;
fi

oldlang=$1
nn_model=$2
vocab=$3 # Vocabulary used for training the neural language model. This is
         # usually the same as $oldlang/words.txt.
data=$4
indir=$5
dir=$6

acwt=$(perl -e "print (1.0/$inv_acwt);")

# Figures out if the old LM is G.fst or G.carpa
oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

for f in $nn_model $vocab $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

nj=$(cat $indir/num_jobs) || exit 1;
mkdir -p $dir
cp $indir/num_jobs $dir/num_jobs

expansion_dir=$dir/lattice_expansion
paths=$dir/paths
scores=$dir/scores
mkdir -p $expansion_dir $paths $scores


if [ $stage -le 0 ]; then
  echo "$0: pruning and expanding lattices."
  $cmd JOB=1:$nj $dir/log/lattice_prune_and_expand.JOB.log \
    lattice-prune --acoustic-scale=$acwt --beam=$beam \
    "ark:gunzip -c $indir/lat.JOB.gz|" ark:- \| \
    lattice-expand --acoustic-scale=$acwt --epsilon=$epsilon \
    ark:- "ark:|gzip -c>$expansion_dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 1 ]; then
  # Convert a lattice into a minimal list of hypotheses under 2 constraints:
  # 1) Every arc on the lattice must be covered by the list.
  # 2) Each hypothesis is the best one for at least one arc it includes.
  echo "$0: converting each lattice into a minimal list of hypotheses."
  $cmd JOB=1:$nj $dir/log/path_cover.JOB.log \
    lattice-path-cover --acoustic-scale=$acwt --word-symbol-table=$oldlang/words.txt \
    "ark:gunzip -c $expansion_dir/lat.JOB.gz|" \
    ark,t:$paths/path.JOB.words ark,t:$paths/states.JOB ark,t:$paths/path_costs.JOB || exit 1;
  for n in $(seq $nj); do
    utils/int2sym.pl -f 2- $oldlang/words.txt < $paths/path.$n.words > $paths/path.$n.words_text || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: computing neural LM scores of the minimal list of hypotheses."
  $cmd JOB=1:$nj $dir/log/compute_sentence_scores.JOB.log \
    PYTHONPATH=steps/pytorchnn python3 steps/pytorchnn/compute_sentence_scores.py \
     --infile $paths/path.JOB.words_text \
     --outfile $scores/${model_type}_scores.JOB \
     --vocabulary $vocab \
     --model-path $nn_model \
     --model $model_type \
     --emsize $embedding_dim \
     --nhid $hidden_dim \
     --nlayers $nlayers \
     --nhead $nhead \
     --oov "$oov_symbol"
fi

if [ $stage -le 3 ]; then
  echo "$0: estimating neural language model scores for each arc."
  $cmd JOB=1:$nj $dir/log/estimate_arc_nnlm_scores.JOB.log \
    python3 steps/pytorchnn/estimate_arc_nnlm_scores.py \
      --arc-ids $paths/states.JOB \
      --scores $scores/${model_type}_scores.JOB \
      --outfile $scores/arc_scores.JOB
fi

# Rescore the expanded lattice: add neural LM scores first and then remove the
# old N-gram LM scores. The two models are effectively interpolated.
oldlm_command="fstproject --project_output=true $oldlm |"
oldlm_weight=$(perl -e "print -1.0 * $weight;")
nnlm_weight=$(perl -e "print $weight;")
if [ $stage -le 4 ]; then
  echo "$0: replaceing old N-gram LM scores with estimated neural language model scores."
  if [ "$oldlm" == "$oldlang/G.fst" ]; then
    $cmd JOB=1:$nj $dir/log/nnlmrescore.JOB.log \
      lattice-add-nnlmscore --lm-scale=$nnlm_weight "ark:gunzip -c $expansion_dir/lat.JOB.gz|" \
      $scores/arc_scores.JOB ark:- \| \
      lattice-lmrescore --lm-scale=$oldlm_weight \
      ark:- "$oldlm_command" "ark:|gzip -c>$dir/lat.JOB.gz" || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/nnlmrescore.JOB.log \
      lattice-add-nnlmscore --lm-scale=$nnlm_weight "ark:gunzip -c $expansion_dir/lat.JOB.gz|" \
      $scores/arc_scores.JOB ark:- \| \
      lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
      ark:- "$oldlm" "ark:|gzip -c>$dir/lat.JOB.gz" || exit 1;
  fi
fi

if ! $skip_scoring ; then
  err_msg="$0: Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $dir
else
  echo "$0: Not scoring because --skip-scoring was specified."
fi

exit 0;
