#!/bin/bash
set -e
set -o pipefail

. path.sh

append_to_orig_feats=true
add_frame_snr=false
nj=4
cmd=run.pl
stage=0
dataid=
compress=true

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-dir> <snr-dir> <tmp-dir> <feat-dir> <out-data-dir>"
  echo " e.g.: $0 data/train_100k_whole_hires exp/frame_snrs_snr_train_100k_whole exp/make_snr_data_dir snr_feats data/train_100k_whole_snr"
  exit 1
fi

data=$1
snr_dir=$2
tmpdir=$3
featdir=$4
dir=$5

extra_files=
$append_to_orig_feats && extra_files="$extra_files $data/feats.scp"
$add_frame_snr && extra_files="$extra_files $snr_dir/frame_snrs.scp"

for f in $snr_dir/nnet_pred_snrs.scp $extra_files; do
  [ ! -f $f ] && echo "$0: Could not find $f" && exit 1 
done

featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

[ -z "$dataid" ] && dataid=`basename $data`
mkdir -p $dir $featdir $tmpdir/$dataid

for n in `seq $nj`; do
  utils/create_data_link.pl $featdir/appended_snr_feats.$n.ark
done

if [ $stage -le 1 ]; then
  if $append_to_orig_feats; then
    utils/split_data.sh $data $nj
    sdata=$data/split$nj

    if $add_frame_snr; then
      append_opts="append-feats ark:- scp:$snr_dir/frame_snrs.scp ark:- |"
    fi

    $cmd JOB=1:$nj $tmpdir/$dataid/make_append_snr_feats.JOB.log \
      append-feats scp:$sdata/JOB/feats.scp scp:$snr_dir/nnet_pred_snrs.scp ark:- \| \
      $append_opts copy-feats --compress=$compress ark:- \
      ark,scp:$featdir/appended_snr_feats.JOB.ark,$featdir/appended_snr_feats.JOB.scp || exit 1
  else
    if $add_frame_snr; then
      append_opts="append-feats scp:- scp:$snr_dir/frame_snrs.scp ark:- |"
    fi

    $cmd JOB=1:$nj $tmpdir/$dataid/make_append_snr_feats.JOB.log \
      utils/split_scp.pl -j $nj \$[JOB-1] $snr_dir/nnet_pred_snrs.scp \| \
      $append_opts copy-feats --compress=$compress ark:- \
      ark,scp:$featdir/appended_snr_feats.JOB.ark,$featdir/appended_snr_feats.JOB.scp || exit 1
  fi

fi

utils/copy_data_dir.sh $data $dir
rm -f $dir/cmvn.scp

for n in `seq $nj`; do
  cat $featdir/appended_snr_feats.$n.scp
done > $dir/feats.scp

