#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)

set -e 
. ./path.sh

cmd=run.pl
nj=
new_lats=
target_data_dir=

help_message="USAGE:$0 [--cmd <run.pl> --nj <nj:default \$1/num_jobs> ] --target_data_dir <target_data_dir> --new_lats <new_lats> <lat1> <lat2> ..."

. ./utils/parse_options.sh

. ./utils/require_argument_all.sh  --new_lats --target_data_dir

[ $# -lt 2 ] && echo "$help_message" && exit 1

[ -d $new_lats ] && echo "ERROR: $new_lats already exist." && exit 1
#utils/validate_data_dir.sh $target_data_dir

if [ -z $nj ] ; then
  [ ! -f $1/num_jobs  ] && echo "$0: expected file $1/num_jobs to exist" && exit 1; 
  nj=$(cat $1/num_jobs)
fi
#utils/split_data.sh $target_data_dir $nj
data=$target_data_dir
sdata=$target_data_dir/split$nj
if [ ! -d $sdata ] ; then
	 echo "$0:Splitting text"
		for j in $(seq 1 $nj); do [ ! -d $sdata/$j ] && mkdir -p $sdata/$j ; done
		utils/split_scp.pl $data/text $(for j in $(seq 1 $nj); do echo -n $sdata/$j/text" " ; done)
fi


mkdir -p $new_lats/log
echo $nj > $new_lats/num_jobs
[ -f $1/final.mdl ] && cp $1/final.mdl $new_lats/
[ -f $1/tree ] && cp $1/tree $new_lats/
[ -f $1/phones.txt ] && cp $1/phones.txt $new_lats/


for old_lats in  $@ ; do
	if [ ! -f $old_lats/lat.scp ] ; then 
		echo "Copy $old_lats/lat.*.gz to lat.ark,lat.scp" 
		old_nj=$(cat $old_lats/num_jobs)
		$cmd JOB=1:$old_nj gunzip -c $old_lats/lat.JOB.gz | lattice-copy ark:- ark,scp:$old_lats/lat.JOB.ark,$old_lats/lat.JOB.scp
		cat $old_lats/lat.*.scp > $old_lats/lat.scp
		cat $old_lats/lat.scp >> $new_lats/lat.scp
	fi
done

$cmd JOB=1:$nj $new_lats/log/subset_lats.JOB.log \
		cat $new_lats/lat.scp \| \
		filter_scp.pl $target_data_dir/split$nj/JOB/text \| sort \| \
		lattice-copy scp:- ark:- \| gzip -c \> $new_lats/lat.JOB.gz
rm $new_lats/lat.scp
echo "Merge lattices: Done" >&2
