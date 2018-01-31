#!/bin/bash

# Copyright    2017  Nagendra Kumar Goel
#              2014  Johns Hopkins University (author: Nagendra K Goel)
# Apache 2.0

# This script operates on a directory, such as in exp/segmentation_1a/train_whole_combined_targets_rev1,
# that contains some subset of the following files:
# targets.X.ark
# frame_subsampling_factor
# It copies to another directory, possibly adding a specified prefix or a suffix
# to the utterance names.


# begin configuration section
utt_prefix=
utt_suffix=
cmd=run.pl
# end configuration section

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <src_dir> <dest_dir>"
  echo "e.g.:"
  echo " $0  --utt-prefix=1- exp/segmentation_1a/train_whole_combined_targets_sub3 exp/segmentation_1a/train_whole_combined_targets_sub3_rev1"
  echo "Options"
  echo "   --utt-prefix=<prefix>     # Prefix for utterance ids, default empty"
  echo "   --utt-suffix=<suffix>     # Suffix for utterance ids, default empty"
  exit 1;
fi


export LC_ALL=C

src_dir=$1
dest_dir=$2

mkdir -p $dest_dir

if [ ! -f $src_dir/targets.1.ark ]; then
  echo "copy_targets_dir.sh: no such files $src_dir/targets.1.ark"
  exit 1;
fi

for f in frame_subsampling_factor; do
  if [ ! -f $src_dir/$f ]; then
    echo "$0: no such file $src_dir/$f this might be serious error."
    continue
  fi
  cp $src_dir/$f $dest_dir/
done

nj=$(ls $src_dir/targets.*.ark | wc -l)
mkdir -p $dest_dir/temp
cat << EOF > $dest_dir/temp/copy_targets.sh
set -e;
id=\$1
echo "$src_dir/targets.\$id.ark"
copy-matrix ark:$src_dir/targets.\$id.ark ark,t:- | \
python -c "
import sys
for line in sys.stdin:
      parts = line.split()
      if \"[\" not in line:
            print line.rstrip()
      else:
            print '$utt_prefix{0}$utt_suffix {1}'.format(parts[0], ' '.join(parts[1:]))
" | \
  copy-matrix ark,t:- ark:$dest_dir/targets.\$id.ark || exit 1;
set +o pipefail; # unset the pipefail option.
EOF
chmod +x $dest_dir/temp/copy_targets.sh
$cmd -v PATH JOB=1:$nj $dest_dir/temp/copy_targets.JOB.log $dest_dir/temp/copy_targets.sh JOB || exit 1;

echo "$0: copied targets from $src_dir to $dest_dir"
