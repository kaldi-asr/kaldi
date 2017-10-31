#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script validates a 'targets_dir' as created by lats_to_targets.sh.
# See that script for details about the format of the targets.

[ -f ./path.sh ] && . ./path.sh

if [ $# -ne 2 ]; then
  cat <<EOF
  This script validates a 'targets_dir' as created by lats_to_targets.sh.
  See that script for details about the format of the targets.

  Usage: steps/segmentation/validate_targets_dir.sh <targets-dir> <data-dir>
  e.g.: steps/segmentation/validate_targets_dir.sh \
    exp/segmentation1a/tri3b_train_split10s_targets \
    data/train_split10s
EOF
  exit 1
fi

targets_dir=$1
data=$2

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT HUP INT PIPE TERM

export LC_ALL=C

function check_sorted_and_uniq {
  ! awk '{print $1}' $1 | sort | uniq | cmp -s - <(awk '{print $1}' $1) && \
    echo "$0: file $1 is not in sorted order or has duplicates" && exit 1;
}

for f in $targets_dir/targets.scp $data/utt2spk; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

utils/data/validate_data_dir.sh --no-text --no-wav --no-spk-sort \
  $data || exit 1

check_sorted_and_uniq $targets_dir/targets.scp

nu=`cat $data/utt2spk | wc -l` || exit 1
nt=`cat $targets_dir/targets.scp | wc -l` || exit 1
if [ $nt -ne $nu ]; then
  echo "WARNING: It seems not all of the targets files were successfully created in "
  echo "$targets_dir/targets.scp for $data ($nt != $nu)."
fi

if [ $nt -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the targets were successfully generated.  Probably a serious error."
  exit 1
fi

head -n 100 $targets_dir/targets.scp | sort -k1,1 | feat-to-len scp:- ark,t:$tmpdir/len.targets || exit 1
utils/filter_scp.pl $tmpdir/len.targets $data/feats.scp | sort -k1,1 | feat-to-len scp:- ark,t:$tmpdir/len.feats || exit 1

frame_subsampling_factor=1
if [ -f $targets_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $targets_dir/frame_subsampling_factor) || exit 1
fi

utils/filter_scp.pl $tmpdir/len.feats $tmpdir/len.targets | \
  paste -d ' ' - $tmpdir/len.feats | python -c "
import sys
num_lines = 0
for line in sys.stdin:
  parts = line.strip().split()
  if parts[0] != parts[2]:
    continue
  len_target = int(parts[1])
  len_feats = int(float(parts[3]) / $frame_subsampling_factor)
  diff = abs(len_target - len_feats)
  if diff > 3:
    sys.stderr.write('Mismatch in length for utterance {utt} between '
                     'targets and feats: {0} vs {1}; diff={2}'.format(
                      len_target, len_feats, diff, utt=parts[0]))
    sys.exit(1)
  num_lines += 1" || exit 1

echo "$0: Successfully validated data-directory $data"
