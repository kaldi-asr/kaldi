#!/bin/bash
local=`pwd`/local

# Create the 'text', 'wav.scp' and utt2spk

python3 $local/parallel2kaldi.py $1 $2 $3 $4

#LC_ALL=C sort $2/"text"
#LC_ALL=C sort $2/"wav.scp"
#LC_ALL=C sort $2/"utt2spk"

utils/utt2spk_to_spk2utt.pl $2/utt2spk > $2/spk2utt

utils/fix_data_dir.sh $2
