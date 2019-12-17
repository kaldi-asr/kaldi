#! /bin/bash 

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Get duration of the utterance given data dir
set -eu
echo $0 $@

data_dir=$1
mkdir -p duration

./utils/data/get_utt2dur.sh $data_dir

echo "$data_dir"
python local/sum_duration.py $data_dir/utt2dur 
echo ""


