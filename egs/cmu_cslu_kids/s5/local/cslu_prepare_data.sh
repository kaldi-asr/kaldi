#! /bin/bash 

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Prepares cslu_kids
# Should be run from egs/cmu_csli_kids

set -e
Looper()
{
    # echo "Looping through $1"
    for f in $1/*; do 
        if [ -d $f ]; then
            Looper $f
        else            
            ./local/cslu_aud_prep.sh --data $data --audio $f
        fi
    done
}

data=data/data_cslu
corpus=cslu
. ./utils/parse_options.sh

rm -f debug/cslu_dataprep_debug
mkdir -p debug
# File check, remove previous data and features files 
for d in $data/test $data/train; do 
    mkdir -p $d
    ./local/file_check.sh $d
done

echo "Preparing cslu_kids..."
Looper $corpus/speech/scripted

for d in $data/test $data/train; do
    ./utils/utt2spk_to_spk2utt.pl $d
    ./utils/fix_data_dir.sh $d
done
if [ -f debug/cslu_dataprep_debug ]; then
    echo "Missing transcripts for some utterances. See cslu_dataprep_debug"
fi

# Optional
# Get data duration, just for book keeping
# for data in data/data_cslu/test data/data_cslu/train; do 
#     ./local/data_duration.sh $data
# done
