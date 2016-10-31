#!/bin/bash
DATA=$1
localdata=data/local
localtmp=$localdata/tmp
find \
    $DATA/ \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name "*sri_gabon_conv_*" | \
    perl -e 'use File::Basename; while (<>) {print basename $_;}' | \
    sort > \
	 $localtmp/sri_gabon_conv_speakers_all.txt
