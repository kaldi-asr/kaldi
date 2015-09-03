#!/bin/bash

# Copyright  2013  Daniel Povey
# Apache 2.0

if [ $# -ne 3 ]; then
  echo "Usage: $0 <fisher-speech> <fisher-transcripts> <out-dir>"
  echo "e.g.: $0 /mnt/data/LDC2004S13 /mnt/data/LDC2004T19 data/train_fisher1"
  echo "or: $0 /mnt/data/LDC2005S13 /mnt/data/LDC2005T19 data/train_fisher2"
  exit 1;
fi

speech=$1
trans=$2
data=$3

tbl1=$trans/fe_03_p1_tran/doc/fe_03_p1_calldata.tbl
tbl2=$trans/fe_03_p2_tran/doc/fe_03_p2_calldata.tbl
if [ -f $tbl1 ]; then
  tbl=$tbl1
elif [ -f $tbl2 ]; then
  tbl=$tbl2
else
  echo "Expecting either $tbl or $tbl2 to exist"
fi

if ! which sph2pipe >/dev/null; then
  echo "$0: sph2pipe is not on your path.";
  exit 1;
fi

if [ ! -d $speech/fe_03_p1_sph1 ] && [ ! -d $speech/fe_03_p2_sph1 ]; then
  echo "$0: expected either directory $speech/fe_03_p1_sph1 or $speech/fe_03_p2_sph1 to exist"
fi

tmpdir=data/local/tmp
mkdir -p $tmpdir || exit 1;
mkdir -p $data || exit 1;
find $speech -name "*.sph" > $tmpdir/sph.list

cmdline="local/make_fisher.pl $tbl $tmpdir/sph.list $data"
if ! $cmdline; then
  echo "$0 Error running command: $cmdline"
  exit 1
fi

utils/utt2spk_to_spk2utt.pl <$data/utt2spk >$data/spk2utt
utils/fix_data_dir.sh $data

exit 0;

