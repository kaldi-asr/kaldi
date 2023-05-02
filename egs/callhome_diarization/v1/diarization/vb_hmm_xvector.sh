#!/usr/bin/env bash

# Copyright       2020  Desh Raj
# Apache 2.0.

# This script performs Bayesian HMM on top of labels produced
# by a first-pass AHC clustering. See https://arxiv.org/abs/1910.08847
# for details about the model.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
rttm_channel=0

# The hyperparameters used here are taken from the DIHARD
# optimal hyperparameter values reported in:
# http://www.fit.vutbr.cz/research/groups/speech/publi/2019/diez_IEEE_ACM_2019_08910412.pdf
# These may require tuning for different datasets.
loop_prob=0.85
fa=0.2
fb=1

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <dir> <xvector-dir> <plda>"
  echo " e.g.: $0 exp/ exp/xvectors_dev exp/xvector_nnet_1a/plda"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

dir=$1
xvec_dir=$2
plda=$3

mkdir -p $dir/tmp

for f in $dir/labels ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# check if numexpr is installed. Also install
# a modified version of kaldi_io with extra functions
# needed to read the PLDA file
result=`python3 -c "\
try:
    import kaldi_io, numexpr
    print (int(hasattr(kaldi_io, 'read_plda')))
except ImportError:
    print('0')"`

if [ "$result" == "0" ]; then
    echo "Installing kaldi_io and numexpr"
    python3 -m pip install git+https://github.com/desh2608/kaldi-io-for-python.git@vbx
    python3 -m pip install numexpr
fi

# Set various variables.
mkdir -p $dir/log

if [ $stage -le 0 ]; then
  # Mean subtraction (If original x-vectors are high-dim, e.g. 512, you should
  # consider also applying LDA to reduce dimensionality to, say, 200) 
  $cmd $xvec_dir/log/transform.log \
    ivector-subtract-global-mean scp:$xvec_dir/xvector.scp \
    ark,scp:$xvec_dir/xvector_norm.ark,$xvec_dir/xvector_norm.scp
fi

cp $xvec_dir/plda_scores/spk2utt $dir/tmp/
cp $xvec_dir/plda_scores/utt2spk $dir/tmp/
cp $xvec_dir/plda_scores/segments $dir/tmp/
utils/fix_data_dir.sh $dir/tmp > /dev/null

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

for n in $(seq 1 $nj); do
  utils/filter_scp.pl $sdata/$n/utt2spk $xvec_dir/xvector_norm.scp > $sdata/$n/xvector.scp
  utils/filter_scp.pl $sdata/$n/utt2spk $dir/labels > $sdata/$n/labels.old
done

if [ $stage -le 1 ]; then
  echo -e "Performing bayesian HMM based x-vector clustering..\n"
  $cmd JOB=1:$nj $dir/log/vb_hmm.JOB.log \
    python3 diarization/vb_hmm_xvector.py \
      --loop-prob $loop_prob --fa $fa --fb $fb \
      $sdata/JOB/xvector.scp $sdata/JOB/spk2utt $plda \
      $sdata/JOB/labels.old $dir/tmp/labels.vb.JOB
fi

if [ $stage -le 2 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/tmp/labels.vb.$j; done > $dir/labels.vb || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: computing RTTM"
  diarization/make_rttm.py --rttm-channel $rttm_channel $xvec_dir/plda_scores/segments $dir/labels.vb $dir/rttm.vb || exit 1;
fi

if $cleanup ; then
  rm -r $dir/tmp || exit 1;
fi
