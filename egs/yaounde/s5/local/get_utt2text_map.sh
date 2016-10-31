#!/bin/bash
localdata=data/local
localtmp=$localdata/tmp
for fld in test train; do
	local/get_utt2text.pl ${localtmp}/${fld}_trans_unsorted.txt > ${localtmp}/${fld}_utt2text_unsorted.txt
        done
