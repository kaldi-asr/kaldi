#!/bin/bash
pth=$1
mkdir -p data/local/lists
find $pth \
     -type f \
     -name "*.trl" \
    | sort \
	  > data/local/lists/trllist.txt
