#!/bin/bash
DATA=$1
localdata=data/local
localtmp=$localdata/tmp
    {
    while read speakeridnumber; do
        speakerdir="${DATA}/$speakeridnumber"
        all_utt2spk_entries=()
        for w in ${speakerdir}/*.wav; do
            wavname=$(basename $w ".wav")
            all_utt2spk_entries+=("${wavname} ${speakeridnumber}")
        done

        for a in "${all_utt2spk_entries[@]}"; do
            echo $a;
        done >> $localtmp/train_utt2spk_unsorted.txt
	    done
    } < $localtmp/speaker_names_train.txt
