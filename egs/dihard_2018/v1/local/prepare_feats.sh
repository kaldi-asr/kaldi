#!/bin/bash
#
# Apache 2.0.

# This script adds deltas, applies sliding window CMVN and writes the features to disk.
#
# Although this kind of script isn't necessary in speaker recognition recipes,
# it can be helpful in the diarization recipes.  The script
# diarization/extract_ivectors.sh extracts i-vectors from very
# short (e.g., 1-2 seconds) segments.  Therefore, in order to apply the sliding
# window CMVN in a meaningful way, it must be performed prior to performing
# the subsegmentation.

nj=40
cmd="run.pl"
stage=0
norm_vars=false
center=true
compress=true
cmn_window=300
delta_window=3
delta_order=2

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_ivector_features"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --norm-vars <true|false>                         # If true, normalize variances in the sliding window cmvn"
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=$(utils/make_absolute.sh $dir)

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $featdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b{14,15,16,17}/$USER/kaldi-data/egs/dihard_2018/v1/ivector-$(date +'%m_%d_%H_%M')/ivector_cmvn_feats/storage $featdir/storage
fi

for n in $(seq $nj); do
  # the next command does nothing unless $featdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $featdir/ivector_cmvn_feats_${name}.${n}.ark
done

cp $data_in/utt2spk $data_out/utt2spk
cp $data_in/spk2utt $data_out/spk2utt
cp $data_in/wav.scp $data_out/wav.scp

write_num_frames_opt="--write-num-frames=ark,t:$featdir/log/utt2num_frames.JOB"

sdata_in=$data_in/split$nj;
utils/split_data.sh $data_in $nj || exit 1;

delta_opts="--delta-window=$delta_window --delta-order=$delta_order"

$cmd JOB=1:$nj $dir/log/create_ivector_cmvn_feats_${name}.JOB.log \
  add-deltas $delta_opts scp:${sdata_in}/JOB/feats.scp ark:- \| \
  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
  ark:- ark:- \| \
  copy-feats --compress=$compress $write_num_frames_opt ark:- \
  ark,scp:$featdir/ivector_cmvn_feats_${name}.JOB.ark,$featdir/ivector_cmvn_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/ivector_cmvn_feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1

for n in $(seq $nj); do
  cat $featdir/log/utt2num_frames.$n || exit 1;
done > $data_out/utt2num_frames || exit 1
rm $featdir/log/utt2num_frames.*

echo "$0: Succeeded creating ivector features for $name"
