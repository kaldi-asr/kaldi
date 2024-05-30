#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


my_name=$(basename $PWD)
dest=$1

mkdir -p  $dest/conf/$my_name/
mkdir -p  $dest/data/
mkdir -p  $dest/exp

for i in data/dev10h.pem data/train data/langp_test/ exp/tri5 exp/tri5_ali lang.conf; do
  tgt=$dest/conf/$my_name/`basename $i`.orig
  [ -f $tgt ] && rm $tgt  
  [ -x $tgt ] && rm $tgt  
  ln -s `readlink -f $i` $tgt
done


(
  set -e -o pipefail
  set -o nounset

  cd $dest
  . ./path.sh
  . ./cmd.sh

  [ ! -d exp/$my_name ] && ln -s ../conf/$my_name/ exp/$my_name || true 
  [ ! -d data/$my_name ] && ln -s ../conf/$my_name/ data/$my_name || true
 
  cd conf/$my_name
  ln -s langp_test.orig lang
  for i in dev10h.pem.orig train.orig; do
    if [ "$i" == "train_sub2.orig" ]; then
      tgt=train
    else
      tgt=${i%.orig}
    fi
    [ -d $tgt ] && rm -r $tgt 
    (cd ../../; utils/copy_data_dir.sh conf/$my_name/$i conf/$my_name/$tgt )
  done

  for i in tri5.orig tri5_ali.orig ; do
    tgt=${i%.orig}
    [ -d $tgt ] && rm -r $tgt
    mkdir -p $tgt
    find -L $i -maxdepth 1 -type f | xargs cp -t $tgt
  done
  cat lang.conf.orig > lang.conf
)
