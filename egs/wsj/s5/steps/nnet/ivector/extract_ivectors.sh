#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  Brno University of Technology (author: Karel Vesely)
# Apache 2.0.


# This script computes iVectors in the same format as extract_ivectors_online.sh,
# except that they are actually not really computed online, they are first computed
# per speaker and just duplicated many times.
# This is mainly intended for use in decoding, where you want the best possible
# quality of iVectors.
#
# This setup also makes it possible to use a previous decoding or alignment, to
# down-weight silence in the stats (default is --silence-weight 0.0).
#
# This is for when you use the "online-decoding" setup in an offline task, and
# you want the best possible results.


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)

posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.

max_count=100       # Interpret this as a number of frames times posterior scale...
                    # this config ensures that once the count exceeds this (i.e.
                    # 1000 frames, or 10 seconds, by default), we start to scale
                    # down the stats, accentuating the prior term.   This seems quite
                    # important for some reason.

silence_weight=0.0
acwt=0.1  # used if input is a decode dir, to get best path from lattices.
mdl=final  # change this if decode directory did not have ../final.mdl present.

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ] && [ $# != 5 ]; then
  echo "Usage: $0 [options] <data> <lang> <extractor-dir> [<alignment-dir>|<decode-dir>|<weights-archive>] <ivector-dir>"
  echo " e.g.: $0 data/test exp/nnet2_online/extractor exp/tri3/decode_test exp/nnet2_online/ivectors_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "                                                   # Ignored if <alignment-dir> or <decode-dir> supplied."
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  echo "  --posterior-scale <float;default=0.1>            # Scale on posteriors in iVector extraction; "
  echo "                                                   # affects strength of prior term."

  exit 1;
fi

set -euxo pipefail

if [ $# -eq 4 ]; then
  data=$1
  lang=$2
  srcdir=$3
  dir=$4
else # 5 arguments
  data=$1
  lang=$2
  srcdir=$3
  ali_or_decode_dir=$4
  dir=$5
fi

for f in $data/feats.scp $srcdir/final.ie $srcdir/final.dubm $lang/phones.txt; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $dir/log
silphonelist=$(cat $lang/phones/silence.csl) || exit 1;

if [ ! -z "$ali_or_decode_dir" ]; then

  if [ -f $ali_or_decode_dir/ali.1.gz ]; then
    if [ ! -f $ali_or_decode_dir/${mdl}.mdl ]; then
      echo "$0: expected $ali_or_decode_dir/${mdl}.mdl to exist."
      exit 1;
    fi
    nj_orig=$(cat $ali_or_decode_dir/num_jobs) || exit 1;

    if [ $stage -le 0 ]; then
      rm $dir/weights.*.gz 2>/dev/null || true

      $cmd JOB=1:$nj_orig  $dir/log/ali_to_post.JOB.log \
        gunzip -c $ali_or_decode_dir/ali.JOB.gz \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $ali_or_decode_dir/final.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c >$dir/weights.JOB.gz" || exit 1;

      # put all the weights in one archive.
      for j in $(seq $nj_orig); do gunzip -c $dir/weights.$j.gz; done | gzip -c >$dir/weights.gz || exit 1;
      rm $dir/weights.*.gz || exit 1;
    fi

  elif [ -f $ali_or_decode_dir/lat.1.gz ]; then
    nj_orig=$(cat $ali_or_decode_dir/num_jobs) || exit 1;
    if [ ! -f $ali_or_decode_dir/../${mdl}.mdl ]; then
      echo "$0: expected $ali_or_decode_dir/../${mdl}.mdl to exist."
      exit 1;
    fi


    if [ $stage -le 0 ]; then
      rm $dir/weights.*.gz 2>/dev/null || true

      $cmd JOB=1:$nj_orig  $dir/log/lat_to_post.JOB.log \
        lattice-best-path --acoustic-scale=$acwt "ark:gunzip -c $ali_or_decode_dir/lat.JOB.gz|" ark:/dev/null ark:- \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $ali_or_decode_dir/../${mdl}.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c >$dir/weights.JOB.gz" || exit 1;

      # put all the weights in one archive.
      for j in $(seq $nj_orig); do gunzip -c $dir/weights.$j.gz; done | gzip -c >$dir/weights.gz || exit 1;
      rm $dir/weights.*.gz || exit 1;
    fi

  elif [ -f $ali_or_decode_dir ] && gunzip -c $ali_or_decode_dir >/dev/null; then
    cp $ali_or_decode_dir $dir/weights.gz || exit 1;

  else
    echo "$0: expected ali.1.gz or lat.1.gz to exist in $ali_or_decode_dir";
    exit 1;
  fi
fi

sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

gmm_feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
feats="$gmm_feats"

# (here originally was the sub-speaker hack),
this_sdata=$sdata

# Per-speaker i-vectors,
if [ $stage -le 2 ]; then
  if [ ! -z "$ali_or_decode_dir" ]; then
    $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      weight-post ark:- "ark,s,cs:gunzip -c $dir/weights.gz|" ark:- \| \
      ivector-extract --acoustic-weight=$posterior_scale --compute-objf-change=true \
        --max-count=$max_count --spk2utt=ark:$this_sdata/JOB/spk2utt \
      $srcdir/final.ie "$feats" ark,s,cs:- ark:$dir/ivectors_spk.JOB.ark
  else
    $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      ivector-extract --acoustic-weight=$posterior_scale --compute-objf-change=true \
        --max-count=$max_count --spk2utt=ark:$this_sdata/JOB/spk2utt \
      $srcdir/final.ie "$feats" ark,s,cs:- ark:$dir/ivectors_spk.JOB.ark
  fi
fi

# Per-utterance i-vectors,
if [ $stage -le 3 ]; then
  if [ ! -z "$ali_or_decode_dir" ]; then
    $cmd JOB=1:$nj $dir/log/extract_ivectors_utt.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      weight-post ark:- "ark,s,cs:gunzip -c $dir/weights.gz|" ark:- \| \
      ivector-extract --acoustic-weight=$posterior_scale --compute-objf-change=true --max-count=$max_count \
      $srcdir/final.ie "$feats" ark,s,cs:- ark:$dir/ivectors_utt.JOB.ark
  else
    $cmd JOB=1:$nj $dir/log/extract_ivectors_utt.JOB.log \
      gmm-global-get-post --n=$num_gselect --min-post=$min_post $srcdir/final.dubm "$gmm_feats" ark:- \| \
      ivector-extract --acoustic-weight=$posterior_scale --compute-objf-change=true --max-count=$max_count \
      $srcdir/final.ie "$feats" ark,s,cs:- ark:$dir/ivectors_utt.JOB.ark
  fi
fi

absdir=$(readlink -f $dir)
if [ $stage -le 4 ]; then
  echo "$0: merging iVectors across jobs"
  copy-vector "ark:cat $dir/ivectors_spk.*.ark |" ark,scp:$absdir/ivectors_spk.ark,$dir/ivectors_spk.scp
  rm $dir/ivectors_spk.*.ark
  copy-vector "ark:cat $dir/ivectors_utt.*.ark |" ark,scp:$absdir/ivectors_utt.ark,$dir/ivectors_utt.scp
  rm $dir/ivectors_utt.*.ark
fi

# duplicate the `speaker' i-vector to all `utterances' of that speaker,
if [ $stage -le 5 ]; then
  utils/apply_map.pl -f 2 $dir/ivectors_spk.scp <$data/utt2spk >$dir/ivectors_spk-as-utt.scp
fi

echo "$0: done extracting iVectors (per-speaker, per-sentence) into '$dir'"

