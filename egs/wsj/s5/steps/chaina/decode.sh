#!/bin/bash

# Copyright 2019  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script does the speaker-dependent pass of decoding with a 'chaina' model,
# including getting the speaker-dependent transforms and dumping lattices.


# Begin configuration section.
stage=1

acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=10.0  # This is typically used in 'chain' systems to scale
                       # acoustics by 10 so the regular scoring script works OK
                       # (since it evaluates the LM scale at integer values,
                       # typically close to 10).  We make this the default in
                       # order to make scoring easier, but you should remember
                       # when using the lattices, that this has been done.
cmd=run.pl
beam=15.0
frames_per_chunk=50
max_active=7000
min_active=200
silence_weight=0.01   # We weight down the posteriors of silence (needs to be tuned).
lattice_beam=6.0 # Beam we use in lattice generation.
iter=final
num_threads=1 # if >1, will use nnet3-latgen-faster-parallel

scoring_opts=
skip_diagnostics=false
skip_scoring=false
# we may later add extra-{left,right}-context options, but these might be
# problematic.
extra_left_context=0
extra_right_context=0
minimize=false
lang=default
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;
set -e -u

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options]  <data-dir> <graph-dir> <model-dir> <embedding-dir> <si-decode-dir> <decode-dir>"
  echo "e.g.:   steps/chaina/decode.sh --nj 8 \\"
  echo "   data/test exp/chaina/tdnn1a_sp/graph_bg exp/chaina/tdnn1a_sp/final"
  echo "   exp/chaina/tdnn1a_sp/data/test exp/chaina/tdnn1a_sp/decode_test_bg.si exp/chaina/tdnn1a_sp/decode_test_bg"
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --lattice-beam <beam>                    # Lattice pruning beam; default 6.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  echo "  --use-gpu <true|false>                   # default: false.  If true, we recommend"
  echo "                                           # to use large --num-threads as the graph"
  echo "                                           # search becomes the limiting factor."
  exit 1;
fi


data=$1
graphdir=$2
model_dir=$3
embedding_dir=$4
si_dir=$5
dir=$6


mkdir -p $dir/log

for f in $graphdir/HCLG.fst $data/utt2spk $model_dir/$lang.mdl $model_dir/$lang.ada \
       $model_dir/info.txt $embedding_dir/output.scp $si_dir/lat.1.gz $si_dir/num_jobs; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


nj=$(cat $si_dir/num_jobs)
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
silphonelist=$(cat $graphdir/phones/silence.csl)
frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$model_dir/info.txt)
bottom_subsampling_factor=$(awk '/^bottom_subsampling_factor/ {print $2}' <$model_dir/info.txt)
top_subsampling_factor=$[frame_subsampling_factor/bottom_subsampling_factor]


## Now get the first-pass fMLLR transforms.
if [ $stage -le 1 ]; then
  echo "$0: getting speaker-dependent transforms"
  # The --acoustic-scale=0.1 is to reverse the --post-decode-acwt (default: 10)
  # that we used when dumping the SI lattices (this was for scoring
  # convenience).
  $cmd JOB=1:$nj $dir/log/get_transform.JOB.log \
    gunzip -c $si_dir/lat.JOB.gz \| \
    lattice-to-post --acoustic-scale=0.1 ark:- ark:- \| \
    weight-silence-post $silence_weight $silphonelist $model_dir/${lang}.mdl ark:- ark:- \| \
    post-to-pdf-post $model_dir/${lang}.mdl ark:- ark:- \| \
    nnet3-adapt --verbose=2 --frame-subsampling-factor=$top_subsampling_factor \
      get-transforms $model_dir/${lang}.ada ark:$sdata/JOB/spk2utt \
      "scp:filter_scp.pl $sdata/JOB/utt2spk $embedding_dir/output.scp|" \
      ark,s,cs:- ark:$dir/trans.JOB.ark
fi

if [ $num_threads -gt 1 ]; then
  thread_string="-parallel --num-threads=$num_threads"
  queue_opt="--num-threads $num_threads"
else
  thread_string=
  queue_opt=
fi

if [ $stage -le 2 ]; then
  $cmd $queue_opt JOB=1:$nj $dir/log/decode.JOB.log \
    nnet3-latgen-faster$thread_string \
     --frame-subsampling-factor=$top_subsampling_factor \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt \
     $model_dir/${lang}.mdl \
     $graphdir/HCLG.fst \
     "ark:filter_scp.pl $sdata/JOB/utt2spk $embedding_dir/output.scp | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB.ark scp:- ark:-|" \
     "ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi


if [ $stage -le 3 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" --model $model_dir/${lang}.mdl $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 4 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi
echo "Decoding done."
exit 0;
