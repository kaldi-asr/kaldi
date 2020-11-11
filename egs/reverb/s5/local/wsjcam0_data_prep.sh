#!/usr/bin/env bash

# Copyright 2013 MERL (author: Felix Weninger)
# Contains some code by Microsoft Corporation, Johns Hopkins University (author: Daniel Povey)

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



dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
root=`pwd`

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

cd $dir

WSJ=$1
wsj0_dir=$2
if [ ! -d "$WSJ" ]; then
    echo Could not find directory $WSJ! Check pathnames in corpus.sh!
    exit 1
fi

# The REVERB Challenge uses only the primary microphone data for the development
# set, but primary and secondary for the evaluation set.
# The REVERB simulated evaluation set (SimData_et) is based on the union of
# the WSJCAM0 si_et_1 and si_et_2 sets. We thus have to merge the transcription,
# script etc. files from si_et_1 and si_et_2 to create the "virtual" si_et set
# so that it can be processed in analogy to si_dt for SimData_dt.

# concatenate dt / et transcription
for set in si_dt si_et_1 si_et_2; do
    # this can be done as in the htk baseline
    if [[ "$set" =~ et ]]; then
        find $WSJ/data/{primary,secondary}_microphone/$set -name '*.wv1' | sort > $set.flist
    else
        find $WSJ/data/primary_microphone/$set -name '*.wv1' | sort > $set.flist
    fi
    nl=`wc -l $set.flist`
    nl=${nl% *}
    echo "$set: $nl files"
    find $WSJ/data/*/$set -type f -name '*.dot' \
        | grep '/[a-z0-9]\{3\}/[a-z0-9]\{3\}c02[a-z0-9]\{2\}\.dot$' \
        | xargs cat > $dir/$set.dot
done
cat $dir/si_et_1.dot $dir/si_et_2.dot > $dir/si_et.dot

# for si_tr we need the transcribed utterances (not all)
si_tr_dot=$WSJ/data/primary_microphone/etc/si_tr.dot
# copy this, for consistency ...
cp $si_tr_dot $dir
chmod 644 $dir/si_tr.dot

utts=$(perl -e 'while (<>) { chomp; if (m/\((\w{8})\)/) { print $1, " "; } }' $si_tr_dot)
for utt in ${utts[@]}; do
   #echo utt = $utt
   spk=${utt:0:3}
   echo $WSJ/data/primary_microphone/si_tr/$spk/$utt.wv1
done > si_tr.flist

nl=`wc -l si_tr.flist`
nl=${nl% *}
echo "si_tr: $nl files"
[ "$nl" -eq 7861 ] || echo "Warning: expected 7861 lines in si_tr.flist, got $nl"

for x in si_tr si_dt si_et_1 si_et_2; do
   $local/flist2scp.pl $x.flist | sort > ${x}_sph.scp
done
cat si_et_{1,2}_sph.scp > si_et_sph.scp

# for WSJCAM0 training set, there's only one transcript file which contains all training speakers
# just use that
$local/convert_transcripts.pl $si_tr_dot > si_tr.trans1 || exit 1

$local/convert_transcripts.pl $dir/si_dt.dot > si_dt.trans1 || exit 1
$local/convert_transcripts.pl $dir/si_et.dot > si_et.trans1 || exit 1


# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in si_tr si_dt si_et; do
   cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort | uniq > $x.txt || exit 1;
done
echo "done" 
# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in si_tr si_dt si_et; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in si_tr si_dt si_et; do
   cat ${x}_sph.scp | awk '{print $1, $1}' > $x.utt2spk
   cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# REVERB language model is bcb05cnp
# We also use tri-gram
echo "Copy language model"
cp $wsj0_dir/wsj0/doc/lng_modl/base_lm/bcb05cnp.z $lmdir/lm_bg_5k.arpa.gz || exit 1;
chmod 644 $lmdir/lm_bg_5k.arpa.gz
cp $wsj0_dir/wsj0/doc/lng_modl/base_lm/tcb05cnp.z $lmdir/lm_tg_5k.arpa.gz || exit 1
chmod 644 $lmdir/lm_tg_5k.arpa.gz

echo "Data preparation succeeded"
