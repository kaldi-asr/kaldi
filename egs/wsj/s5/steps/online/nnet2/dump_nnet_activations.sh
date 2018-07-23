#!/bin/bash

# Copyright   2013  Daniel Povey
# Apache 2.0.

# This script was modified from ./extract_ivectors_online2.sh.  It is to be used
# when retraining the top layer of a system that was trained on another,
# out-of-domain dataset, on some in-domain dataset.  It takes as input a
# directory such as nnet_gpu_online as prepared by ./prepare_online_decoding.sh,
# and a data directory, and it processes the wave files to get features and iVectors,
# then puts it through all but the last layer of the neural net in that directory, and dumps
# those final activations in a feats.scp file in the output directory.  These files
# might be quite large.  A typical feature-dimension is 300; it's the p-norm output dim.
# We compress these files (note: the compression is lossy).


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
utts_per_spk_max=2 # maximum 2 utterances per "fake-speaker."

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <srcdir> <output-dir>"
  echo " e.g.: $0 data/train exp/nnet2_online/nnet_a_online exp/nnet2_online/activations_train"
  echo "Output is in <output-dir>/feats.scp"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue-opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --utts-per-spk-max <int;default=2>    # Controls splitting into 'fake speakers'."
  echo "                                        # Set to 1 if compatibility with utterance-by-utterance"
  echo "                                        # decoding is the only factor, and to larger if you care "
  echo "                                        # also about adaptation over several utterances."
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

for f in $data/wav.scp $srcdir/conf/online_nnet2_decoding.conf $srcdir/final.mdl; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
echo $nj >$dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;


mkdir -p $dir/conf $dir/feats
grep -v '^--endpoint' $srcdir/conf/online_nnet2_decoding.conf > $dir/conf/online_feature_pipeline.conf

if [ $stage -le 0 ]; then
  ns=$(wc -l <$data/spk2utt)
  if [ "$ns" == 1 -a "$utts_per_spk_max" != 1 ]; then
    echo "$0: you seem to have just one speaker in your database.  This is probably not a good idea."
    echo "  see http://kaldi-asr.org/doc/data_prep.html (search for 'bold') for why"
    echo "  Setting --utts-per-spk-max to 1."
    utts_per_spk_max=1
  fi

  mkdir -p $dir/spk2utt_fake
  for job in $(seq $nj); do 
   # create fake spk2utt files with reduced number of utterances per speaker,
   # so the network is well adapted to using iVectors from small amounts of
   # training data.
    awk -v max=$utts_per_spk_max '{ n=2; count=0; while(n<=NF) {
      nmax=n+max; count++; printf("%s-%06x", $1, count); for (;n<nmax&&n<=NF; n++) printf(" %s", $n); print "";} }' \
        <$sdata/$job/spk2utt >$dir/spk2utt_fake/spk2utt.$job
  done
fi

if [ $stage -le 1 ]; then
  info=$dir/nnet_info
  nnet-am-info $srcdir/final.mdl >$info
  nc=$(grep num-components $info | awk '{print $2}');
  if grep SumGroupComponent $info >/dev/null; then 
    nc_truncate=$[$nc-3]  # we did mix-up: remove AffineComponent,
                          # SumGroupComponent, SoftmaxComponent
  else
    nc_truncate=$[$nc-2]  # remove AffineComponent, SoftmaxComponent
  fi
  nnet-to-raw-nnet --truncate=$nc_truncate $srcdir/final.mdl $dir/nnet.raw
fi

if [ $stage -le 2 ]; then
  echo "$0: dumping neural net activations"

  # The next line is a no-op unless $dir/feats/storage/ exists; see utils/create_split_dir.pl.
  for j in $(seq $nj); do  utils/create_data_link.pl $dir/feats/feats.$j.ark; done

  if [ -f $data/segments ]; then
    wav_rspecifier="ark,s,cs:extract-segments scp,p:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- |"
  else
    wav_rspecifier="scp,p:$sdata/JOB/wav.scp"
  fi
  $cmd JOB=1:$nj $dir/log/dump_activations.JOB.log \
    online2-wav-dump-features  --config=$dir/conf/online_feature_pipeline.conf \
      ark:$dir/spk2utt_fake/spk2utt.JOB "$wav_rspecifier" ark:- \| \
    nnet-compute $dir/nnet.raw ark:- ark:- \| \
    copy-feats --compress=true ark:- \
      ark,scp:$dir/feats/feats.JOB.ark,$dir/feats/feats.JOB.scp || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: combining activations across jobs"
  mkdir -p $dir/data
  cp -r $data/* $dir/data
  for j in $(seq $nj); do cat $dir/feats/feats.$j.scp; done >$dir/data/feats.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: computing [fake] CMVN stats."
  # We shouldn't actually be doing CMVN, but the get_egs.sh script expects it,
  # so create fake CMVN stats.
  steps/compute_cmvn_stats.sh --fake $dir/data $dir/log $dir/feats || exit 1
fi


echo "$0: done.  Output is in $dir/data/feats.scp"
