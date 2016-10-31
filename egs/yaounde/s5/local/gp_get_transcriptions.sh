#!/bin/bash
fld=$1
localdata=data/local
localtmp=$localdata/tmp
{
    while read speakernumber; do
	$(cat data/prompts/${speakernumber}/prompts >> ${localtmp}/${fld}_trans_unsorted.txt)
        done
    } < $localtmp/${fld}_speakers_all.txt
