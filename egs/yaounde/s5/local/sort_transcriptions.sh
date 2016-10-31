#!/bin/bash
fld=$1
localdata=data/local
localtmp=$localdata/tmp
outdir=data
sort -u  ${localtmp}/${fld}_utt2text_unsorted.txt > ${outdir}/${fld}/filename2text
sort -u  ${localtmp}/${fld}_utt2text_unsorted.txt > ${outdir}/${fld}/text
sort  -u $localtmp/${fld}_wav_unsorted.scp > ${outdir}/${fld}/wav.scp
sort -u  $localtmp/${fld}_spk2utt_unsorted.txt > ${outdir}/${fld}/spk2utt
sort -u  $localtmp/${fld}_utt2spk_unsorted.txt > ${outdir}/${fld}/utt2spk
