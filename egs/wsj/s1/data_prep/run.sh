# This script should be run from its own directory (.)
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# It takes as arguments a list of directories that should end
# with numbers like 13-4.1.  These are the subdirectories in the WSJ disks.
# on the BUT system we can get these by doing:
#  ./run.sh /mnt/matylda2/data/WSJ?/??-{?,??}.?

# Another example is:
#  ./run.sh  /ais/gobi2/speech/WSJ/*/??-{?,??}.?


if [ $# -lt 4 ]; then
   echo "Too few arguments to run.sh: need a list of WSJ directories ending e.g. 11-13.1"
   exit 1;
fi

rm -r links/ 2>/dev/null
mkdir links/
ln -s $* links

# This version for SI-84

cat links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 ./ndx2flist.pl $* | sort | \
 grep -v 11-2.1/wsj0/si_tr_s/401 > train_si84.flist

# This version for SI-284
cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx \
 links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 ./ndx2flist.pl  $* | sort | \
 grep -v 11-2.1/wsj0/si_tr_s/401 > train_si284.flist


# Now for the test sets.
# links/13-34.1/wsj1/doc/indices/readme.doc 
# describes all the different test sets.
# Note: each test-set seems to come in multiple versions depending
# on different vocabulary sizes, verbalized vs. non-verbalized
# pronunciations, etc.  We use the largest vocab and non-verbalized
# pronunciations.
# The most normal one seems to be the "baseline 60k test set", which
# is h1_p0. 

# Nov'92 (333 utts)
# These index files have a slightly different  format;
# have to add .wv1
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  ./ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > eval_nov92.flist

# Nov'93: (213 utts)
# Have to replace a wrong disk-id.
cat links/13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
  sed s/13_32_1/13_33_1/ | \
  ./ndx2flist.pl $* | sort > eval_nov93.flist

# Dev-set for Nov'93 (503  utts)
cat links/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  ./ndx2flist.pl $* | sort > dev_nov93.flist

# Dev-set for Nov'93 (503 utts)
# links/13-34.1/wsj1/doc/indices/h1_p0.ndx

# Finding the transcript files:
for x in $*; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for x in train_si84 train_si284 eval_nov92 eval_nov93 dev_nov93; do
   ./flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' | ./find_transcripts.pl  dot_files.flist > $x.trans1
done

# Do some initial normalization steps.
noiseword="<NOISE>";
for x in train_si84 train_si284 eval_nov92 eval_nov93 dev_nov93; do
   cat $x.trans1 | ./normalize_transcript.pl $noiseword > $x.trans2 || exit 1
done

if [ ! -f ../data/lexicon.txt ]; then
   echo  "You need to get ../data/lexicon.txt first (see ../run.sh)"
   exit 1
fi
# Convert OOVs to <SPOKEN_NOISE>
spoken_noise_word="<SPOKEN_NOISE>";
for x in train_si84 train_si284 eval_nov92 eval_nov93 dev_nov93; do
   cat $x.trans2 | ./oov2unk.pl ../data/lexicon.txt $spoken_noise_word | sort  > $x.txt  || exit 1 # the .txt is the final transcript.
done
 
# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
sph2pipe=`cd ../../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
if [ ! -f $sph2pipe ]; then
   echo "Could not find the sph2pipe program at $sph2pipe";
   exit 1;
fi
for x in train_si84 train_si284 eval_nov92 eval_nov93 dev_nov93; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done


# The 20K vocab, open-vocabulary language model (i.e. the one with UNK), without
# verbalized pronunciations.   This is the most common test setup, I understand.

cp links/13-32.1/wsj1/doc/lng_modl/base_lm/bcb20onp.z  lm_bg.arpa.gz
chmod u+w lm_bg.arpa.gz
# trigram would be:

cat links/13-32.1/wsj1/doc/lng_modl/base_lm/tcb20onp.z | \
 perl -e 'while(<>){ if(m/^\\data\\/){ print; last;  } } while(<>){ print; }' | \
 gzip -c -f > lm_tg.arpa.gz

export PATH=$PATH:../../../../tools/irstlm/bin
prune-lm --threshold=1e-7 lm_tg.arpa.gz lm_tg_pruned.arpa
gzip -f lm_tg_pruned.arpa

# Make the utt2spk and spk2utt files.
for x in train_si84 train_si284 eval_nov92 eval_nov93 dev_nov93; do
   cat ${x}_sph.scp | awk '{print $1}' | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
   cat $x.utt2spk | ../scripts/utt2spk_to_spk2utt.pl > $x.spk2utt
done


if [ ! -f wsj0-train-spkrinfo.txt ]; then
  wget http://www.ldc.upenn.edu/Catalog/docs/LDC93S6A/wsj0-train-spkrinfo.txt
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
# disks.  I put it in the repository.

cat links/11-13.1/wsj0/doc/spkrinfo.txt \
    links/13-34.1/wsj1/doc/train/spkrinfo.txt \
   ./wsj0-train-spkrinfo.txt  | \
   perl -ane 'tr/A-Z/a-z/;print;' | grep -v ';' | \
   awk '{print $1, $2}' | grep -v -- -- > spk2gender.map

