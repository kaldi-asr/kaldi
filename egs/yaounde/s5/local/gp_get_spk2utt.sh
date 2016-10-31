#!/bin/bash
DATA=$1
fld=$2
localdata=data/local
localtmp=$localdata/tmp
{
    while read speakeridnumber; do
	speakerdir="${DATA}/$speakeridnumber"
        all_spk2utt_entries=()
	            all_spk2utt_entries+=("${speakeridnumber} ")
        for w in ${speakerdir}/*.wav; do
            wavname=$(basename $w ".wav")
            all_spk2utt_entries+=("${wavname}")
        done

        for a in "${all_spk2utt_entries[@]}"; do
            echo -n "$a ";
        done >> $localtmp/${fld}_spk2utt_unsorted.txt
    echo "" >> $localtmp/${fld}_spk2utt_unsorted.txt
    done
} < $localtmp/${fld}_speakers_all.txt
