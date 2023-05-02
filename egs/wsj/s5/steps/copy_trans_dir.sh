#!/usr/bin/env bash
# Copyright 2019   Phani Sankar Nidadavolu
# Copyright 2019   manhong wang(marvin)
# Apache 2.0.

#This script creates fmllr transform for the aug dirs by copying 
#the trans of original train dir after you copy_ali_dirs.sh or copy_lat_dirs.sh
#Note :  wo do not accept --nj here ,which shoud keep same as ali file
prefixes="reverb1 babble music noise"
include_original=true
cmd=run.pl
write_binary=true

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <out-data> <src-ali-dir> <out-ali-dir>"
  echo "This script creates fmllr transform for the aug dirs by copying "
  echo " the trans of original train dir"
  echo "While copying it adds prefix to the utterances specified by prefixes option"
  echo "Note that the original train dir does not have any prefix"
  echo "To include the original training directory in the copied "
  echo "version set the --include-original option to true"
  echo "main options (for others, see top of script file)"
  echo "  --prefixes <string of prefixes to add>    # All the prefixes of aug data to be included"
  echo "  --include-original <true/false>           # If true, will copy the alignements of original dir"
  exit 1
fi

data=$1
src_dir=$2
dir=$3

if [ ! -d $dir ]; then
    echo "$0: warning : you may need combine ali or lat first !" && exit 1
fi

if [ ! -f $src_dir/trans.1 ] ; then
    echo "$0: no trans exist in $src_dir dir"  && exit 1
fi


nj=$(cat $dir/num_jobs)
rm -f $dir/trans* 2>/dev/null

# Copy the fmllr trans temporarily
echo "creating temporary trans in $dir"
$cmd  JOB=1:$nj $dir/log/copy_trans_temp.JOB.log \
  copy-matrix --binary=$write_binary \
  "ark:cat $src_dir/trans.JOB |" \
  ark,scp:$dir/trans_tmp.JOB.ark,$dir/trans_tmp.JOB.scp || exit 1

# Make copies of utterances for perturbed data
for p in $prefixes; do
  cat $dir/trans_tmp.*.scp | awk -v p=$p '{print p"-"$0}'
done | sort -k1,1 > $dir/trans_out.scp.aug

if [ "$include_original" == "true" ]; then
  cat $dir/trans_tmp.*.scp | awk '{print $0}' | sort -k1,1 > $dir/trans_out.scp.clean
  cat $dir/trans_out.scp.clean $dir/trans_out.scp.aug | sort -k1,1 > $dir/trans_out.scp.old
else
  cat $dir/trans_out.scp.aug | sort -k1,1 > $dir/trans_out.scp.old
fi

utils/filter_scp.pl  ${data}/spk2utt  $dir/trans_out.scp.old  >  $dir/trans_out.scp
utils/split_data.sh ${data} $nj

# Copy and dump the trans for perturbed data
echo Creating fmllr trans for augmented data by copying fmllr trans from clean data
$cmd  JOB=1:$nj $dir/log/copy_out_trans.JOB.log \
  copy-matrix --binary=$write_binary \
  "scp:utils/split_scp.pl  --one-based -j $nj JOB $dir/trans_out.scp |" \
  ark:$dir/trans.JOB || exit 1

n_aug_trans=`wc -l $data/spk2utt`
n_copy_trans=`wc -l $dir/trans_out.scp`
echo "copy $n_copy_trans speaker's  fmllr trans of total $n_aug_trans"
rm $dir/trans_out.scp.aug  $dir/trans_out.scp.old $dir/trans_out.scp   $dir/trans_tmp.*
exit 0
