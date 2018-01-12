#!/bin/bash
# Copyright 2017 Nagendra Kumar Goel
# Apache 2.0.

# This srcipt operates on targets directories, such as exp/segmentation_1a/train_whole_combined_targets_sub3
# the output is a new targets dir which has targets from all the input targets dirs

# Begin configuration section.
cmd=run.pl
extra_files=
num_jobs=4
# End configuration section.
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 [options] <data> <dest-targets-dir> <src-targets-dir1> <src-targets-dir2> ..."
  echo "e.g.: $0 --num-jobs 32 data/train exp/targets_combined exp/targets_1 exp/targets_2"
  echo "Options:"
  echo " --extra-files <file1 file2...>   # specify addtional files in 'src-targets-dir1' to copy"
  echo " --num-jobs <nj>                  # number of jobs used to split the data directory."
  echo " Note, files that don't appear in the first source dir will not be added even if they appear in later ones."
  echo " Other than alignments, only files from the first src ali dir are copied."
  exit 1;
fi

data=$1;
shift;
dest=$1;
shift;
first_src=$1;

mkdir -p $dest;
rm $dest/{targets.*.ark,frame_subsampling_factor} 2>/dev/null

cp $first_src/frame_subsampling_factor $dest 2>/dev/null

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/targets.1.ark ]; then
    echo "$0: check if targets (targets.*.ark) are present in $dir."
    exit 1;
  fi
done

for dir in $*; do
  for f in frame_subsampling_factor; do
    diff $first_src/$f $dir/$f 1>/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "$0: Cannot combine alignment directories with different $f files."
    fi
  done
done

for f in frame_subsampling_factor $extra_files; do
  if [ ! -f $first_src/$f ]; then
    echo "combine_targets_dir.sh: no such file $first_src/$f"
    exit 1;
  fi
  cp $first_src/$f $dest/
done

src_id=0
temp_dir=$dest/temp
[ -d $temp_dir ] && rm -r $temp_dir;
mkdir -p $temp_dir
echo "$0: dumping targets in each source directory as single archive and index."
for dir in $*; do
  src_id=$((src_id + 1))
  cur_num_jobs=$(ls $dir/targets.*.ark | wc -l) || exit 1;
  tgts=$(for n in $(seq $cur_num_jobs); do echo -n "$dir/targets.$n.ark "; done)
  $cmd $dir/log/copy_targets.log \
    copy-matrix "ark:cat $tgts|" \
    ark,scp:$temp_dir/targets.$src_id.ark,$temp_dir/targets.$src_id.scp || exit 1;
done
sort -m $temp_dir/targets.*.scp > $dest/targets.scp || exit 1;


echo "Combined targets and stored in $dest"
exit 0
