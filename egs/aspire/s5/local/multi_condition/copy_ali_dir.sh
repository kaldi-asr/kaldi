#!/usr/bin/env bash

# Copyright 2014  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0

# This script operates on a directory, such as in exp/tri4a_ali,
# that contains some subset of the following files:
#  ali.*.gz
#  tree
#  cmvn_opts
#  splice_opts
#  num_jobs
#  final.mdl
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
  echo " $0  --utt-prefix=1- exp/tri4a_ali exp/tri4a_rev1_ali"
  echo "Options"
  echo "   --utt-prefix=<prefix>     # Prefix for utterance ids, default empty"
  echo "   --utt-suffix=<suffix>     # Suffix for utterance ids, default empty"
  exit 1;
fi


export LC_ALL=C

src_dir=$1
dest_dir=$2

mkdir -p $dest_dir

if [ ! -f $src_dir/ali.1.gz ]; then
  echo "copy_ali_dir.sh: no such files $src_dir/ali.*.gz"
  exit 1;
fi

for f in tree cmvn_opts splice_opts num_jobs final.mdl; do
  if [ ! -f $src_dir/$f ]; then
    echo "copy_ali_dir.sh: no such file $src_dir/$f this might be serious error."
    continue
  fi
  cp $src_dir/$f $dest_dir/
done

nj=$(cat $dest_dir/num_jobs)
mkdir -p $dest_dir/temp
cat << EOF > $dest_dir/temp/copy_ali.sh
set -e;
id=\$1
echo "$src_dir/ali.\$id.gz"
gunzip -c $src_dir/ali.\$id.gz | \
  copy-int-vector ark:- ark,t:- | \
python -c "
import sys
for line in sys.stdin:
  parts = line.split()
  print '$utt_prefix{0}$utt_suffix {1}'.format(parts[0], ' '.join(parts[1:]))
" | \
  gzip -c >$dest_dir/ali.\$id.gz || exit 1;
set +o pipefail; # unset the pipefail option.
EOF
chmod +x $dest_dir/temp/copy_ali.sh
$cmd -v PATH JOB=1:$nj $dest_dir/temp/copy_ali.JOB.log $dest_dir/temp/copy_ali.sh JOB || exit 1;

echo "$0: copied alignments from $src_dir to $dest_dir"
