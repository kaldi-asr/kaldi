#!/bin/bash
DATA=$1
localdata=data/local
localtmp=$localdata/tmp
find $DATA/ -mindepth 1 -maxdepth 1 -type d | perl -e 'use File::Basename; while (<>) {print basename $_;}' | sort -u > $localtmp/speakers_all.txt
