#!/bin/bash

cat $1 | awk '{sub(/M=/,"",$2); sub(/G=.*/,"eps",$2); print $2}' | uniq | awk 'BEGIN{ cnt=0; } {print $1,cnt; cnt=cnt+1}' > $1.sym