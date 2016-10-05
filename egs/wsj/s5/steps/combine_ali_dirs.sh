#!/bin/bash
# Copyright 2016  Xiaohui Zhang  Apache 2.0.

# This srcipt operates on alignment directories, such as exp/tri4a_ali
# the output is a new ali dir which has alignments from all the input ali dirs

# Begin configuration section.
cmd=run.pl
extra_files=
num_jobs=4
# End configuration section.
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 [options] <data> <dest-ali-dir> <src-ali-dir1> <src-ali-dir2> ..."
  echo "e.g.: $0 --num-jobs 32 data/train exp/tri3_ali_combined exp/tri3_ali_1 exp_tri3_ali_2"
  echo "Options:"
  echo " --extra-files <file1 file2...>   # specify addtional files in 'src-ali-dir1' to copy"
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
rm $dest/{ali.*.gz,num_jobs} 2>/dev/null

cp $first_src/phones.txt $dest 2>/dev/null

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/ali.1.gz ]; then
    echo "$0: check if alignments (ali.*.gz) are present in $dir."
    exit 1;
  fi
done

for dir in $*; do
  for f in tree; do
    diff $first_src/$f $dir/$f 1>/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "$0: Cannot combine alignment directories with different $f files."
    fi
  done
done

for f in final.mdl tree cmvn_opts num_jobs $extra_files; do
  if [ ! -f $first_src/$f ]; then
    echo "combine_ali_dir.sh: no such file $first_src/$f"
    exit 1;
  fi
  cp $first_src/$f $dest/
done

src_id=0
temp_dir=$dest/temp
[ -d $temp_dir ] && rm -r $temp_dir;
mkdir -p $temp_dir
echo "$0: dumping alignments in each source directory as single archive and index."
for dir in $*; do
  src_id=$((src_id + 1))
  cur_num_jobs=$(cat $dir/num_jobs) || exit 1;
  alis=$(for n in $(seq $cur_num_jobs); do echo -n "$dir/ali.$n.gz "; done)
  $cmd $dir/log/copy_alignments.log \
    copy-int-vector "ark:gunzip -c $alis|" \
    ark,scp:$temp_dir/ali.$src_id.ark,$temp_dir/ali.$src_id.scp || exit 1;
done
sort -m $temp_dir/ali.*.scp > $temp_dir/ali.scp || exit 1;

echo "$0: splitting data to get reference utt2spk for individual ali.JOB.gz files."
utils/split_data.sh $data $num_jobs || exit 1;

echo "$0: splitting the alignments to appropriate chunks according to the reference utt2spk files."
utils/filter_scps.pl JOB=1:$num_jobs \
  $data/split$num_jobs/JOB/utt2spk $temp_dir/ali.scp $temp_dir/ali.JOB.scp

for i in `seq 1 $num_jobs`; do
    copy-int-vector scp:$temp_dir/ali.${i}.scp "ark:|gzip -c >$dest/ali.$i.gz" || exit 1;
done

echo $num_jobs > $dest/num_jobs  || exit 1

echo "$0: checking the alignment files generated have at least 90% of the utterances."
for i in `seq 1 $num_jobs`; do
  num_lines=`cat $temp_dir/ali.$i.scp | wc -l` || exit 1;
  num_lines_tot=`cat $data/split$num_jobs/$i/utt2spk | wc -l` || exit 1;
  python -c "import sys;
percent = 100.0 * float($num_lines) / $num_lines_tot
if percent < 90 :
  print ('$dest/ali.$i.gz {0}% utterances missing.'.format(percent))"  || exit 1;
done
rm -r $temp_dir 2>/dev/null

echo "Combined alignments and stored in $dest"
exit 0
