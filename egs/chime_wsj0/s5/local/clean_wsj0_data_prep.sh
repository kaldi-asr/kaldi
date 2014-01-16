#!/bin/bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 29/05/12

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level WSJ corpus directory."
  echo "It is assumed that there will be a 'wsj0' and a 'wsj1' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

CORPUS=$1

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

cd $dir

# This version for SI-84
cat $CORPUS/wsj0/doc/indices/train/tr_s_wv1.ndx \
  | $local/cstr_ndx2flist.pl $CORPUS | sort -u > train_si84_clean.flist

# This version for SI-284
#cat $CORPUS/wsj1/doc/indices/si_tr_s.ndx \
#  $CORPUS/wsj0/doc/indices/train/tr_s_wv1.ndx \
#  | $local/cstr_ndx2flist.pl  $CORPUS | sort \
#  | grep -v wsj0/si_tr_s/401 > train_si284.flist

# Now for the test sets.
# $CORPUS/wsj1/doc/indices/readme.doc 
# describes all the different test sets.
# Note: each test-set seems to come in multiple versions depending
# on different vocabulary sizes, verbalized vs. non-verbalized
# pronunciations, etc.  We use the largest vocab and non-verbalized
# pronunciations.
# The most normal one seems to be the "baseline 60k test set", which
# is h1_p0. 

# Nov'92 (333 utts)
# These index files have a slightly different format; 
# have to add .wv1, which is done in cstr_ndx2flist.pl 
cat $CORPUS/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  $local/cstr_ndx2flist.pl $CORPUS | sort > test_eval92_clean.flist

# Nov'92 (330 utts, 5k vocab)
cat $CORPUS/wsj0/doc/indices/test/nvp/si_et_05.ndx | \
  $local/cstr_ndx2flist.pl $CORPUS | sort > test_eval92_5k_clean.flist

# Nov'93: (213 utts)
# Have to replace a wrong disk-id.
#cat $CORPUS/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
#  $local/cstr_ndx2flist.pl $CORPUS | sort > test_eval93.flist

# Nov'93: (215 utts, 5k)
#cat $CORPUS/wsj1/doc/indices/wsj1/eval/h2_p0.ndx | \
#  $local/cstr_ndx2flist.pl $CORPUS | sort > test_eval93_5k.flist

# Dev-set for Nov'93 (503 utts)
#cat $CORPUS/wsj1/doc/indices/h1_p0.ndx | \
#  $local/cstr_ndx2flist.pl $CORPUS | sort > test_dev93.flist

# Dev-set for Nov'93 (513 utts, 5k vocab)
#cat $CORPUS/wsj1/doc/indices/h2_p0.ndx | \
#  $local/cstr_ndx2flist.pl $CORPUS | sort > test_dev93_5k.flist


# Dev-set Hub 1,2 (503, 913 utterances)

# Note: the ???'s below match WSJ and SI_DT, or wsj and si_dt.  
# Sometimes this gets copied from the CD's with upcasing, don't know 
# why (could be older versions of the disks).
find $CORPUS/wsj0/si_dt_20 -print | grep -i ".wv1" | sort > dev_dt_20_clean.flist
find $CORPUS/wsj0/si_dt_05 -print | grep -i ".wv1" | sort > dev_dt_05_clean.flist


# Finding the transcript files:
find -L $CORPUS -iname '*.dot' > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
# adding suffix to utt_id 
# 0 for clean condition
for x in train_si84_clean test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean; do
  $local/flist2scp.pl $x.flist | sort > ${x}_sph_tmp.scp
  cat ${x}_sph_tmp.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > ${x}_tmp.trans1
  cat ${x}_sph_tmp.scp | awk '{printf("%s0 %s\n", $1, $2);}' > ${x}_sph.scp
  cat ${x}_tmp.trans1 | awk '{printf("%s0 ", $1); for(i=2;i<=NF;i++) printf("%s ", $i); printf("\n");}' > ${x}.trans1
done




# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si84_clean test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean; do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done
 
# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si84_clean test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp \
    > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train_si84_clean test_eval92_clean test_eval92_5k_clean dev_dt_05_clean dev_dt_20_clean; do
  cat ${x}_sph.scp | awk '{print $1}' \
    | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

#in case we want to limit lm's on most frequent words, copy lm training word frequency list
cp $CORPUS/wsj0/doc/lng_modl/vocab/wfl_64.lst $lmdir
chmod u+w $lmdir/*.lst # had weird permissions on source.

# The 20K vocab, open-vocabulary language model (i.e. the one with UNK), without
# verbalized pronunciations.   This is the most common test setup, I understand.

cp $CORPUS/wsj0/doc/lng_modl/base_lm/bcb20onp.z $lmdir/lm_bg.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg.arpa.gz

# trigram would be:
cat $CORPUS/wsj0/doc/lng_modl/base_lm/tcb20onp.z | \
  perl -e 'while(<>){ if(m/^\\data\\/){ print; last;  } } while(<>){ print; }' \
  | gzip -c -f > $lmdir/lm_tg.arpa.gz || exit 1;

prune-lm --threshold=1e-7 $lmdir/lm_tg.arpa.gz $lmdir/lm_tgpr.arpa || exit 1;
gzip -f $lmdir/lm_tgpr.arpa || exit 1;

# repeat for 5k language models
cp $CORPUS/wsj0/doc/lng_modl/base_lm/bcb05onp.z  $lmdir/lm_bg_5k.arpa.gz || exit 1;
chmod u+w $lmdir/lm_bg_5k.arpa.gz

# trigram would be: !only closed vocabulary here!
cp $CORPUS/wsj0/doc/lng_modl/base_lm/tcb05cnp.z $lmdir/lm_tg_5k.arpa.gz || exit 1;
chmod u+w $lmdir/lm_tg_5k.arpa.gz
gunzip $lmdir/lm_tg_5k.arpa.gz
tail -n 4328839 $lmdir/lm_tg_5k.arpa | gzip -c -f > $lmdir/lm_tg_5k.arpa.gz
rm $lmdir/lm_tg_5k.arpa

prune-lm --threshold=1e-7 $lmdir/lm_tg_5k.arpa.gz $lmdir/lm_tgpr_5k.arpa || exit 1;
gzip -f $lmdir/lm_tgpr_5k.arpa || exit 1;


if [ ! -f wsj0-train-spkrinfo.txt ] || [ `cat wsj0-train-spkrinfo.txt | wc -l` -ne 134 ]; then
  rm -f wsj0-train-spkrinfo.txt
  wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt \
    || ( echo "Getting wsj0-train-spkrinfo.txt from backup location" && \
         wget --no-check-certificate https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt );
fi

if [ ! -f wsj0-train-spkrinfo.txt ]; then
  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
  echo "This is possibly omitted from the training disks; couldn't find it." 
  echo "Everything else may have worked; we just may be missing gender info"
  echo "which is only needed for VTLN-related diagnostics anyway."
  exit 1
fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

cat $CORPUS/wsj0/doc/spkrinfo.txt \
    ./wsj0-train-spkrinfo.txt  | \
    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
    awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender


echo "Data preparation succeeded"
