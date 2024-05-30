#!/usr/bin/env bash

#
# Copyright 2014 Univercity of Tehran (Author: Bagher BabaAli)
#           2014 Brno University of Technology (Karel Vesely)
#           2014 Johns Hopkins University (Daniel Povey)
#
# farsdat, description of the database:
# http://www.assta.org/sst/SST-94-Vol-ll/cache/SST-94-VOL2-Chapter15-p20.pdf

if [ $# -ne 1 ]; then
   echo "Argument should be the farsdat directory, see ../run.sh for example."
   exit 1;
fi

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin

[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";
[ -f $conf/train_spk.list ] || error_exit "$PROG: train-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased
if [ ! -d $*/CD1 -o ! -d $*/CD2 ] && [ ! -d $*/cd1 -o ! -d $*/cd2 ]; then
  echo "farsdat_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to Farsdat directory"
  echo "with name like /export/corpora5/ELRA/farsdat"
  exit 1;
fi

# Now check what case the directory structure is
uppercased=false
cd1_dir=cd1
cd2_dir=cd2
if [ -d $*/CD1 ]; then
  uppercased=true
  cd1_dir=CD1
  cd2_dir=CD2
fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

find $*/{$cd1_dir/SENTENCE/,$cd2_dir/SENTENCE/} -iname '*.SNT' -print |\
  while read filename; do
    rec_id=$(echo "$filename" | sed -e 's:.*/S\([1-2]\)\(.*\)\.SNT$:\2 \1:i' |\
      awk '{printf("%03d_%d\n",$1,$2);}' ) || exit 1;
    cat "$filename" | awk -v rec_id=$rec_id \
      '{printf "%s_%s %s %f %f\n",rec_id,$1,rec_id,$2/(2*22050),$3/(2*22050)}'
  done > $dir/segments || exit 1;

find $*/{$cd1_dir/wave,$cd2_dir/wave} -iname '*.WAV' -print > $tmpdir/wav.flist || exit 1;
sed -e 's:.*/S\([1-2]\)\(.*\)\.WAV$:\2 \1:i' $tmpdir/wav.flist |\
  awk '{printf("%03d_%d\n",$1,$2);}' > $tmpdir/wav.uttids || exit 1;

paste $tmpdir/wav.uttids $tmpdir/wav.flist | \
  awk '{printf("%s sox %s -t wav -r 16000 -c 1 - |\n", $1, $2);}' | sort -k1,1 > $dir/wav.scp

  # Now, Convert the transcripts into our format (no normalization yet)
  # Get the transcripts: each line of the output contains an utterance 
  # ID followed by the transcript.

find $*/{$cd1_dir/PHONEME,$cd2_dir/PHONEME} -iname 'PH*.*' -print > $tmpdir/phn.flist
sed -e 's:.*/PH\([1-2]\)\(.*\)\.\(.*\)$:\2 \1 \3:i' $tmpdir/phn.flist |\
  awk '{printf("%03d_%d_%d\n",$1,$2,$3);}' > $tmpdir/phn.uttids || exit 1;

while read line; do
  [ -f $line ] || error_exit "Cannot find transcription file '$line'";
  cut -c1 "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;' || exit 1;
done < $tmpdir/phn.flist > $tmpdir/phn.trans || exit 1;

paste $tmpdir/phn.uttids $tmpdir/phn.trans | sort -k1,1 > $dir/trans || exit 1;

# Do normalization steps. 
$local/farsdat_norm_trans.sh $dir/trans | sort > $dir/text || exit 1;

 # Prepare gender mapping
cat $*/$cd1_dir/Information/Speaker.txt $*/$cd2_dir/Information/Speaker.txt | \
sed '/Code/d' | awk '{printf("%03d %s\n",$1,$3)}' > $dir/spk2gender || exit 1;

for x in dev test; do
  cat $conf/${x}_spk.list | awk '{printf("%03d\n",$1);}' > \
    $tmpdir/${x}_spk.list || exit 1;
  awk -F'_' 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/${x}_spk.list $dir/segments |\
    sort -k1 | awk -F'_'  '{sent[$1]=sent[$1] " " $3 } 
                          END { 
                                for(i=1; i<=304; ++i)
                                { split(sent[i],sent_split," "); 
                                  asort(sent_split,sent_sort); 
                                  for(j=1; j<=8;j++)
                                  { 
                                    print sent_sort[j];
                                  }
                                }
                              }' | sort -n | uniq > $tmpdir/${x}.sent || exit 1; 
done 

cat $conf/train_spk.list | awk '{printf("%03d\n",$1);}' > \
  $tmpdir/train_spk.list|| exit 1;
cat $tmpdir/dev.sent $tmpdir/test.sent | uniq -u > $tmpdir/dev+test.sent|| exit 1;
seq 1 404 | sed  '/400/d' | grep -F -x -v -f $tmpdir/dev+test.sent - > \
  $tmpdir/train.sent || exit 1;

for x in train dev test; do
  set=data/$x
  mkdir -p $set

  awk -F'_' 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/${x}_spk.list $dir/segments |\
    sort -k1 > $tmpdir/segments || exit 1;

  awk -F'_' 'NR==FNR{a[$1]++;next} (a[substr($3,1,index($3," ")-1)])' \
    $tmpdir/${x}.sent $tmpdir/segments | sort -k1  > $set/segments || exit 1;

  awk -F'_' 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/${x}_spk.list $dir/text |\
    sort -k1 > $tmpdir/text || exit 1;

  awk -F'_' 'NR==FNR{a[$1]++;next} (a[substr($3,1,index($3," ")-1)])' \
    $tmpdir/${x}.sent $tmpdir/text | sort -k1 > $set/text || exit 1;

  awk -F'_' 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/${x}_spk.list $dir/wav.scp > \
  $tmpdir/wav.scp || exit 1;

  cat $set/segments | awk -F'_' '{printf("%03d_%d\n",$1,$2)}' > \
    $tmpdir/spk_session || exit 1;

  awk -F' ' 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/spk_session $tmpdir/wav.scp |\
    sort -k1 > $set/wav.scp || exit 1;

  awk 'NR==FNR{a[$1]++;next} (a[$1])' $tmpdir/${x}_spk.list $dir/spk2gender |\
    tr '[:upper:]' '[:lower:]' > $set/spk2gender || exit 1;

  # Make the utt2spk and spk2utt files.
  cut -d' ' -f1 $set/segments | awk -F'_' '{print $0,$1}'  > $set/utt2spk || exit 1;
  cat $set/utt2spk | utils/utt2spk_to_spk2utt.pl > $set/spk2utt || exit 1;

  # Prepare STM file for sclite:
  awk -v txt=$set/text -v sex=$set/spk2gender \
  'BEGIN{ 
     while(getline < txt) { ref[$1]=substr($0, index($0,$2)); } 
     while(getline < sex) { gender[$1]=$2; } 
     print ";; LABEL \"O\" \"Overall\" \"Overall\"";
     print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
     print ";; LABEL \"M\" \"Male\" \"Male speakers\""; 
   } 
   { spk_id=substr($2,1,3);    
     printf("%s 1 %s %s %s <O,%s> %s\n", $1, spk_id, $3, $4, toupper(gender[spk_id]), ref[$1]);
   }' $set/segments >$set/stm || exit 1

  # Create dummy GLM file for sclite:
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > $set/glm 

  # Check that data dirs are okay!
  utils/validate_data_dir.sh --no-feats $set || exit 1

done

echo "Data preparation succeeded"
