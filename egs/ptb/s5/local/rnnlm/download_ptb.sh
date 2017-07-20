#!/bin/bash

. path.sh
dir=data/rnnlm
data_dir=$dir/data
mkdir -p $data_dir
echo "*** Downloading PTB text data ***"
(
 data_path=https://raw.githubusercontent.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/master/data/
 cd $data_dir
 [ ! -f $data_dir/ptb.txt ] && wget -nv $data_path/ptb.train.txt || exit 1;
 mv ptb.train.txt ptb.txt 
 [ ! -f $data_dir/dev.txt ] && wget -nv $data_path/ptb.valid.txt || exit 1;
 mv ptb.valid.txt dev.txt
 cd ..
 if [ ! -d test ]; then
   mkdir -p test
   cd test
   wget -nv $data_path/ptb.test.txt || exit 1;
 fi
)
echo "*** Finished downloading PTB text data ***"

