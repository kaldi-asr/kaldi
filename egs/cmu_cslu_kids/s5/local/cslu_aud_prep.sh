#/bin/bash 

# Copyright Johns Hopkins University
#   2019 Fei Wu

# Called by local/cslu_DataPrep.shi

Assignment()
{
    rnd=$((1+RANDOM % 100))
    if [ $rnd -le $test_percentage ]; then 
        target="test"
    else
        target="train"
    fi
}
audio=
test_percentage=30  # Percent of data reserved as test set 
debug=debug/cslu_dataprep_debug
data=data/data_cslu
. ./utils/parse_options.sh

uttID=$(basename $audio)
uttID=${uttID%'.wav'}
sentID=${uttID: -3}
spkID=${uttID%$sentID}
sentID=${sentID%"0"}
sentID=$(echo "$sentID" | tr '[:lower:]' '[:upper:]' )

line=$(grep $sentID cslu/docs/all.map)

if [ -z "$line" ]; then     # Can't map utterance to transcript
    echo $audio $sentID >> $debug
else
    txt=$(echo $line | grep -oP '"\K.*?(?=")')
    cap_txt=${txt^^}
    Assignment
    echo "$uttID $cap_txt" >> $data/$target/text
    echo "$uttID $spkID" >> $data/$target/utt2spk
    echo "$spkID f" >> $data/$target/spk2gender
    echo "$uttID $audio" >> $data/$target/wav.scp
fi

