#!/bin/bash

# Copyright     2013  Daniel Povey
# Apache 2.0.

# This script is as ./extract_ivectors_online.sh but internally it uses a
# different program, with code that corresponds more closely to the real online
# decoding setup.  Rather than treating each utterance separately, as
# extract_ivectors_online.sh, it carries forward information from one utterance
# to the next, within the speaker.  However, take note of the option
# "utts-per-spk-max", defaulting to 2, which splits speakers up into "fake
# speakers" with at most two utterances in them.  This means that more iVectors
# are estimated starting from an uninformative starting point, than if we used
# the real speaker labels (which may have many utterances each); it's a
# compromise between per-utterance and per-speaker iVector estimation.


# This script is based on ^/egs/sre08/v1/sid/extract_ivectors.sh.  Instead of
# extracting a single iVector per utterance, it extracts one every few frames
# (controlled by the --ivector-period option, e.g. 10, which is to save compute).
# This is used in training (and not-really-online testing) of neural networks
# for online decoding.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

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
utts_per_spk_max=2  # maximum 2 utterances per "fake-speaker."  Note: this does
                    # not have to be an integer; if it's noninteger, it will be
                    # rounded in a randomized way to one of the two integers it's
                    # close to.  This is useful in the "perturbed-feature" recipe
                    # to encourage that different perturbed versions of the same
                    # speaker get split into fake-speakers differently.
compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).

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
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  echo "  --utts-per-spk-max <int;default=2>    # Controls splitting into 'fake speakers'."
  echo "                                        # Set to 1 if compatibility with utterance-by-utterance"
  echo "                                        # decoding is the only factor, and to larger if you care "
  echo "                                        # also about adaptation over several utterances."
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

for f in $data/feats.scp $srcdir/final.ie $srcdir/final.dubm $srcdir/global_cmvn.stats $srcdir/splice_opts \
     $srcdir/online_cmvn.conf $srcdir/final.mat; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log $dir/conf

sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

echo $ivector_period > $dir/ivector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)

# the program ivector-extract-online2 does a bunch of stuff in memory and is
# config-driven...  this was easier in this case because the same code is
# involved in online decoding.  We need to create a config file for iVector
# extration.

ieconf=$dir/conf/ivector_extractor.conf
echo -n >$ieconf
cp $srcdir/online_cmvn.conf $dir/conf/ || exit 1;
echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$ieconf
for x in $(echo $splice_opts); do echo "$x"; done > $dir/conf/splice.conf
echo "--splice-config=$dir/conf/splice.conf" >>$ieconf
echo "--lda-matrix=$srcdir/final.mat" >>$ieconf
echo "--global-cmvn-stats=$srcdir/global_cmvn.stats" >>$ieconf
echo "--diag-ubm=$srcdir/final.dubm" >>$ieconf
echo "--ivector-extractor=$srcdir/final.ie" >>$ieconf
echo "--num-gselect=$num_gselect"  >>$ieconf
echo "--min-post=$min_post" >>$ieconf
echo "--posterior-scale=$posterior_scale" >>$ieconf
echo "--max-remembered-frames=1000" >>$ieconf # the default



ns=$(wc -l <$data/spk2utt)
if [ "$ns" == 1 -a "$utts_per_spk_max" != 1 ]; then
  echo "$0: you seem to have just one speaker in your database.  This is probably not a good idea."
  echo "  see http://kaldi.sourceforge.net/data_prep.html (search for 'bold') for why"
  echo "  Setting --utts-per-spk-max to 1."
  utts_per_spk_max=1
fi

mkdir -p $dir/spk2utt_fake
for job in $(seq $nj); do 
   # create fake spk2utt files with reduced number of utterances per speaker,
   # so the network is well adapted to using iVectors from small amounts of
   # training data.
   # the if (rand() % 2 == 0)
   awk -v max=$utts_per_spk_max '{ n=2; count=0;
      while(n<=NF) {
        int_max=int(max)+ (rand() < (max-int(max))?1:0); print int_max; 
        nmax=n+int_max; count++; printf("%s-%06x", $1, count); 
        for (;n<nmax&&n<=NF; n++) printf(" %s", $n); print "";} }' \
    <$sdata/$job/spk2utt >$dir/spk2utt_fake/spk2utt.$job
done


for n in $(seq $nj); do
  # This will do nothing unless the directorys $dir/storage exists;
  # it can be used to distribute the data among multiple machines.
  utils/create_data_link.pl $dir/ivector_online.$n.ark
done

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
     ivector-extract-online2 --config=$ieconf ark:$dir/spk2utt_fake/spk2utt.JOB scp:$sdata/JOB/feats.scp ark:- \| \
     copy-feats --compress=$compress ark:- \
      ark,scp,t:$dir/ivector_online.JOB.ark,$dir/ivector_online.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi
