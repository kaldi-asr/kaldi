#! /bin/bash

set -e 

if [ $# -ne 3 ]; then
  echo "Usage: split_data_on_reco.sh <ref-data-dir> <data-dir> <nj>"
  exit 1
fi

ref_data=$1
data=$2
nj=$3

utils/data/get_reco2utt.sh $ref_data
utils/data/get_reco2utt.sh $data

utils/split_data.sh --per-reco $ref_data $nj
 
for n in `seq $nj`; do 
  srn=$ref_data/split${nj}reco/$n
  dsn=$data/split${nj}reco/$n
  
  mkdir -p $dsn

  utils/data/get_reco2utt.sh $srn
  utils/filter_scp.pl $srn/reco2utt $data/reco2utt > $dsn/reco2utt
  utils/spk2utt_to_utt2spk.pl $dsn/reco2utt > $dsn/utt2reco 
  utils/subset_data_dir.sh --utt-list $dsn/utt2reco $data $dsn
done
