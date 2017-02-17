#!/bin/bash

# Copyright     2013  Daniel Povey
# Apache 2.0.

set -o pipefail

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# The script is based on ^/egs/sre08/v1/sid/extract_ivectors.sh.  Instead of
# extracting a single iVector per utterance, it extracts one every few frames
# (controlled by the --ivector-period option, e.g. 10, which is to save compute).
# This is used in training (and not-really-online testing) of neural networks
# for online decoding.

# Rather than treating each utterance separately, it carries forward
# information from one utterance to the next, within the speaker.


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
ivector_period=10
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.
compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).
max_count=0         # The use of this option (e.g. --max-count 100) can make
                    # iVectors more consistent for different lengths of
                    # utterance, by scaling up the prior term when the
                    # data-count exceeds this value.  The data-count is after
                    # posterior-scaling, so assuming the posterior-scale is 0.1,
                    # --max-count 100 starts having effect after 1000 frames, or
                    # 10 seconds of data.
# End configuration section.

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <extractor-dir> <ivector-dir>"
  echo " e.g.: $0 data/train exp/nnet2_online/extractor exp/nnet2_online/ivectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

for f in $data/feats.scp $srcdir/final.ie $srcdir/final.dubm $srcdir/global_cmvn.stats $srcdir/splice_opts \
     $srcdir/online_cmvn.conf $srcdir/final.mat; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log $dir/conf
num_cmn_offsets=-1  # If > 0, it uses offsets.scp and generates iVector for
                    # all random offsets version of inputs.
if [ -f $data/offsets.scp ]; then
  num_cmn_offsets=`feat-to-len --print-args=false scp:"head -n 1 $data/offsets.scp |"`
fi

echo $num_cmn_offsets > $dir/num_cmn_offsets || exit 1;

ivector_prefix=""
if [ $num_cmn_offsets -gt -1 ];then
  ivector_prefix="_offset"
fi

sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
#utils/split_data.sh $data $nj || exit 1;

echo $ivector_period > $dir/ivector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)

# the program ivector-extract-online2 does a bunch of stuff in memory and is
# config-driven...  this was easier in this case because the same code is
# involved in online decoding.  We need to create a config file for iVector
# extraction.

ieconf=$dir/conf/ivector_extractor.conf
echo -n >$ieconf
cp $srcdir/online_cmvn.conf $dir/conf/ || exit 1;
echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$ieconf
for x in $(echo $splice_opts); do echo "$x"; done > $dir/conf/splice.conf
echo "--ivector-period=$ivector_period" >>$ieconf
echo "--splice-config=$dir/conf/splice.conf" >>$ieconf
echo "--lda-matrix=$srcdir/final.mat" >>$ieconf
echo "--global-cmvn-stats=$srcdir/global_cmvn.stats" >>$ieconf
echo "--diag-ubm=$srcdir/final.dubm" >>$ieconf
echo "--ivector-extractor=$srcdir/final.ie" >>$ieconf
echo "--num-gselect=$num_gselect"  >>$ieconf
echo "--min-post=$min_post" >>$ieconf
echo "--posterior-scale=$posterior_scale" >>$ieconf
echo "--max-remembered-frames=1000" >>$ieconf # the default
echo "--max-count=$max_count" >>$ieconf


absdir=$(readlink -f $dir)

for n in $(seq $nj); do
  # This will do nothing unless the directory $dir/storage exists;
  # it can be used to distribute the data among multiple machines.
  utils/create_data_link.pl $dir/ivector_online.$n.ark
done

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  offset_opts=
  if [ -f $data/offsets.scp ]; then
    for i in $(seq $nj); do
      split_feats=$sdata/$i
      awk '{print $1,$2}' < $split_feats/spk2utt | utils/apply_map.pl -f 2 $split_feats/offsets.scp > $split_feats/spk_offsets.scp
    done
    offset_opts="--spk2cmn-offset=scp:$sdata/JOB/spk_offsets.scp"
  fi
  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
     ivector-extract-online2 --config=$ieconf $offset_opts ark:$sdata/JOB/spk2utt scp:$sdata/JOB/feats.scp ark:- \| \
     copy-feats --compress=$compress ark:- \
      ark,scp:$absdir/ivector_online${ivector_perfix}.JOB.ark,$absdir/ivector_online${ivector_prefix}.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online${ivector_prefix}.$j.scp; done >$dir/ivector_online${ivector_prefix}.scp || exit 1;
fi

if [ $num_cmn_offsets -gt 0 ]; then
  echo "$0: generate two separate ivector scp files for ivector with no offsets "
  echo "$dir/ivector_online.scp and ivectors with offset $dir/ivector_online.scp."
  ivec_dim=`feat-to-dim --print-args=false scp:$dir/ivector_online_offset.scp  - || exit 1`
  orig_ivec_dim=$[$ivec_dim/$num_cmn_offsets]
  awk -v ivec_orig_end_col=$[$orig_ivec_dim-1] '{print $0"[:,0:"ivec_orig_end_col"]"}' $dir/ivector_online_offset.scp > $dir/ivector_online.scp
fi
