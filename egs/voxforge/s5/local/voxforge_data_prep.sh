#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

# Makes train/test splits

source path.sh

echo "=== Starting initial VoxForge data preparation ..."

echo "--- Making test/train data split ..."

# The number of speakers in the test set
nspk_test=30

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <data-directory>";
  exit 1;
fi

command -v flac >/dev/null 2>&1 ||\
 { echo "FLAC decompressor needed but not found"'!' ; exit 1; }

DATA=$1

locdata=data/local
loctmp=$locdata/tmp
rm -rf $loctmp >/dev/null 2>&1
mkdir -p $locdata
mkdir -p $loctmp
# The "sed" expression below is quite messy because some of the directrory
# names don't follow the "speaker-YYYYMMDD-<random_3letter_suffix>" convention.
# The ";tx;d;:x" part of the expression is to filter out the directories,
# not matched by the expression
find $DATA/* -maxdepth 0 |\
 sed -Ee 's:.*/((.+)\-[0-9]{8,10}[a-z]*([_\-].*)?):\2:;tx;d;:x' |\
 sort -u > $loctmp/speakers_all.txt

nspk_all=`wc -l $loctmp/speakers_all.txt | cut -f1 -d' '`
if [ "$nspk_test" -ge "$nspk_all" ]; then
  echo "${nspk_test} test speakers requested, but there are only ${nspk_all} speakers in total!"
  exit 1;
fi

utils/shuffle_list.pl <$loctmp/speakers_all.txt | head -n $nspk_test | sort -u >$loctmp/speakers_test.txt

gawk 'NR==FNR{spk[$0]; next} !($0 in spk)' \
    $loctmp/speakers_test.txt $loctmp/speakers_all.txt |\
  sort -u > $loctmp/speakers_train.txt

wc -l $loctmp/speakers_all.txt
wc -l $loctmp/speakers_{train,test}.txt

# expand speaker names to their respective directories
ls -d ${DATA}/*/ |\
 while read d; do  basename $d; done |\
 gawk 'BEGIN {FS="-"} NR==FNR{arr[$1]; next;} ($1 in arr)' \
  $loctmp/speakers_test.txt - | sort > $loctmp/dir_test.txt

ls -d ${DATA}/*/ |\
 while read d; do  basename $d; done |\
 gawk 'BEGIN {FS="-"} NR==FNR{arr[$1]; next;} ($1 in arr)' \
  $loctmp/speakers_train.txt - | sort > $loctmp/dir_train.txt

logdir=exp/data_prep
mkdir -p $logdir
> $logdir/make_trans.log
rm ${locdata}/spk2gender 2>/dev/null
for s in test train; do
 echo "--- Preparing ${s}_wav.scp, ${s}_trans.txt and ${s}.utt2spk ..." 
 while read d; do
  spkname=`echo $d | cut -f1 -d'-'`;
  spksfx=`echo $d | cut -f2- -d'-'`; # | sed -e 's:_:\-:g'`;
  idpfx="${spkname}-${spksfx}";
  dir=${DATA}/$d

  rdm=`find $dir/etc/ -iname 'readme'`
  if [ -z $rdm ]; then
    echo "No README file for $d - skipping this directory ..."
    continue
  fi
  spkgender=`sed -e 's:.*Gender\:[^[:alpha:]]\+\(.\).*:\L\1:gi;tx;d;:x' <$rdm`
  if [ $spkgender != "f" -a $spkgender != "m" ]; then
    echo "Illegal or empty gender ($spkgender) for \"$d\" - assuming m(ale) ..."
    spkgender="m"
  fi
  echo "$spkname $spkgender" >> $locdata/spk2gender.tmp
  
  if [ ! -f ${dir}/etc/PROMPTS ]; then
   echo "No etc/PROMPTS file exists in $dir - skipping the dir ..." \
     >> $logdir/make_trans.log
   continue
  fi
  
  if [ -d ${dir}/wav ]; then
   wavtype=wav
  else
   if [ -d ${dir}/flac ]; then
    wavtype=flac
   else
    echo "No 'wav' or 'flac' dir in $dir - skipping ..."
    continue
   fi
  fi

  all_wavs=""
  while read w; do
    bw=`basename $w`
    wavname=${bw%.$wavtype}
    all_wavs="$all_wavs $wavname"
    id="${idpfx}-${wavname}"
    if [ ! -s $w ]; then
     echo "$w is zero-size - skipping ..."
     continue
    fi
    if [ $wavtype == "wav" ]; then
     echo "$id $w" >> ${loctmp}/${s}_wav.scp.unsorted
    else
     echo "$id flac -c -d --silent $w |" >> ${loctmp}/${s}_wav.scp.unsorted
    fi
    echo "$id $spkname" >> $loctmp/${s}.utt2spk.unsorted
  done < <( ls ${dir}/${wavtype}/*${wavtype} )
  
  local/make_trans.py $dir/etc/PROMPTS ${idpfx} "${all_wavs}" \
   2>>${logdir}/make_trans.log >> ${loctmp}/${s}_trans.txt.unsorted
 done < $loctmp/dir_${s}.txt

 # filter out the audio for which there is no proper transcript
 gawk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
   ${loctmp}/${s}_trans.txt.unsorted ${loctmp}/${s}_wav.scp.unsorted |\
   sort -k1 > ${locdata}/${s}_wav.scp
 
 gawk 'NR==FNR{trans[$1]; next} ($1 in trans)' FS=" " \
   ${loctmp}/${s}_trans.txt.unsorted $loctmp/${s}.utt2spk.unsorted |\
   sort -k1 > ${locdata}/${s}.utt2spk
 
 sort -k1 < ${loctmp}/${s}_trans.txt.unsorted > ${locdata}/${s}_trans.txt

 echo "--- Preparing ${s}.spk2utt ..."
 cat $locdata/${s}_trans.txt |\
  cut -f1 -d' ' |\
  gawk 'BEGIN {FS="-"}
        {names[$1]=names[$1] " " $0;}
        END {for (k in names) {print k, names[k];}}' | sort -k1 > $locdata/${s}.spk2utt
done;

trans_err=`wc -l ${logdir}/make_trans.log | cut -f1 -d" "`
if [ "${trans_err}" -ge 1 ]; then
  echo -n "$trans_err errors detected in the transcripts."
  echo " Check ${logdir}/make_trans.log for details!" 
fi

gawk '{spk[$1]=$2;} END{for (s in spk) print s " " spk[s]}' \
  $locdata/spk2gender.tmp | sort -k1 > $locdata/spk2gender

echo "*** Initial VoxForge data preparation finished!"
