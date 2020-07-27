#!/usr/bin/env bash

#Simple script to create compound set info that will allow for more automatized
#work with the shadow set.
#
#The notion of shadow data set came from the need to be able to verify
#the output of the recognizer during decoding the evaluation data.
#The idea is simple -- instead of decoding just the eval data, decode both
#eval data plus the dev data (or at least some portion of it) interleved
#randomly
#After decoding, we can isolate (split) the output from the decoding (and kws)
#so that we can score the dev data subset and if the score is identical to
#the score obtained by decoding the dev set previously, we can be little bit
#more sure that the eval set results are correct.

. ./path.sh

[ ! -f lang.conf ] && echo "File lang.conf must exist (and contain a valid config)"
. ./lang.conf

devset=dev10h.uem
evlset=eval.uem
tgtset=shadow.uem
tgtdir=

. utils/parse_options.sh
[ -z $tgtdir ] && tgtdir=data/$tgtset

devset_basename=${devset%%.*}
devset_segments=${devset#*.}

evlset_basename=${evlset%%.*}
evlset_segments=${evlset#*.}

eval devset_flist=\$${devset_basename}_data_list
eval devset_ecf=\$${devset_basename}_ecf_file
eval devset_rttm=\$${devset_basename}_rttm_file
eval devset_stm=\$${devset_basename}_stm_file

eval evlset_flist=\$${evlset_basename}_data_list
eval evlset_ecf=\$${evlset_basename}_ecf_file

rm -rf $tgtdir/compounds
mkdir -p $tgtdir/compounds
mkdir -p $tgtdir/compounds/$devset
mkdir -p $tgtdir/compounds/$evlset

echo "Creating compound $tgtdir/compounds/$devset"
(
  cd $tgtdir/compounds/$devset
  echo "DEVSET file list: $devset_flist"
  ln -s `utils/make_absolute.sh $devset_flist` files.list
  echo "DEVSET ECF file : $devset_ecf"
  ln -s `utils/make_absolute.sh $devset_ecf` ecf.xml
  echo "DEVSET RTTM file: $devset_rttm"
  ln -s `utils/make_absolute.sh $devset_rttm` rttm
  echo "DEVSET STM file : $devset_stm"
  ln -s `utils/make_absolute.sh $devset_stm` stm
)

echo "Creating compound $tgtdir/compounds/$evlset"
(
  cd $tgtdir/compounds/$evlset
  echo "EVLSET file list: $evlset_flist"
  ln -s `utils/make_absolute.sh $evlset_flist` files.list
  echo "EVLSET ECF file : $evlset_ecf"
  ln -s `utils/make_absolute.sh $evlset_ecf` ecf.xml
)

echo "Compound creation OK."


