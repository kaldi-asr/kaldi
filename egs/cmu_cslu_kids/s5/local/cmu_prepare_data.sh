#! /bin/bash

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Prepares cmu_kids. 
# Should be run from egs/cmu_cslu_kids

set -eu
corpus=cmu_kids/kids
data=data/data_cmu
test_percentage=30

. ./path.sh
. ./utils/parse_options.sh

total_cnt=0
test_cnt=0
train_cnt=0

for d in $data/train $data/test; do
    mkdir -p $d
    ./local/file_check.sh $d
done

echo "Preparing cmu_kids..."
for kid in $corpus/*; do 
	if [ -d $kid ]; then
        # echo "Kid: $kid"
		spkID=$(basename $kid)
		sph="$kid/signal"
	    if [ -d $sph ];then
            # echo "$sph"
            for utt in $sph/*; do
                if [ ${utt: -4} == ".sph" ]; then
                    total_cnt=$[$total_cnt+1]   
                    rnd=$((1+RANDOM % 100))
                    uttID=$(basename $utt)
                    uttID=${uttID%".sph"}
                    sentID=${uttID#$spkID}
                    sentID=${sentID:0:3}

                    # Find the sentence
                    grep $sentID cmu_kids/tables/sentence.tbl > tmp
                    cut -f 3- < tmp > out                    
        
                    tr '[:lower:]' '[:upper:]' < out > tmp
                    tr -d '[:cntrl:]' < tmp > out
                    sent=$(<out)

                    # Clean transcript 
                    cp $kid/trans/$uttID.trn tmp
                    tr -d '\n' < tmp > out
                    tr '[:lower:]' '[:upper:]' < tmp > out
                    trans=$(<out)
                     
                    if [ $rnd -le $test_percentage ]; then
                        target="test"
                        test_cnt=$[$test_cnt+1]
                    else
                        target="train"
                        train_cnt=$[$train_cnt+1]
                    fi

                    echo "$uttID $spkID" >> $data/$target/utt2spk
                    echo "$uttID $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 $utt|" >> $data/$target/wav.scp
                    echo "$spkID f" >> $data/$target/spk2gender
                    echo "$uttID $sent" >> $data/$target/text
                fi
            done
        fi
	fi
done

for d in $data/train $data/test; do
    utils/utt2spk_to_spk2utt.pl $d/utt2spk > $d/spk2utt
    utils/fix_data_dir.sh $d
done

printf "\t total: %s; train: %s; test: %s.\n" "$total_cnt" "$train_cnt" "$test_cnt" 
rm -f out tmp

# Optional
# Get data duration, just for book keeping
# for data in $data/train $data/test; do
#     ./local/data_duration.sh $data
# done
# 

