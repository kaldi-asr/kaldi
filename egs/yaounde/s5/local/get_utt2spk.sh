#!/bin/bash
DATA=$1
fld=$2
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
        done >> $localtmp/${fld}_utt2spk_unsorted.txt
	    done
} < $localtmp/${fld}_speaker_names.txt
