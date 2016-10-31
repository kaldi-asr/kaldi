#!/bin/bash
# assumes train and test folds exist for gp and yaounde
# under ../gp/data/local/tmp and
# ../yaounde/data/local/tmp
for fld in test train; do
cat ../gp/data/local/tmp/speaker_directory_paths_${fld}.txt ../yaounde/data/local/tmp/speaker_directory_paths_${fld}.txt >> data/local/tmp/speaker_directory_paths_${fld}.txt
done
