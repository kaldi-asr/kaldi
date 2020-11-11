#!/usr/bin/env bash
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
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
# limitations under the License. #

# The vystadial data are specific by having following marks in transcriptions
# _INHALE_
# _LAUGH_
# _EHM_HMM_
# _NOISE_
# _EHM_HMM_
# _SIL_

locdict=$1; shift

echo "--- Prepare nonsilence phone lists ..."
# We suppose only nonsilence_phones in lexicon right now
awk '{for(n=2;n<=NF;n++) { p[$n]=1; }} END{for(x in p) {print x}}' \
    $locdict/lexicon.txt | sort > $locdict/nonsilence_phones.txt

echo "--- Adding silence phones to lexicon ..."
echo "_SIL_ SIL" >> $locdict/lexicon.txt
echo "_EHM_HMM_ EHM" >> $locdict/lexicon.txt
echo "_INHALE_ INH" >> $locdict/lexicon.txt
echo "_LAUGH_ LAU" >> $locdict/lexicon.txt
echo "_NOISE_ NOI" >> $locdict/lexicon.txt

echo "--- Sorting lexicon in place..."
sort $locdict/lexicon.txt -o $locdict/lexicon.txt

echo "--- Prepare silence phone lists ..."
echo SIL > $locdict/silence_phones.txt
echo EHM >> $locdict/silence_phones.txt
echo INH >> $locdict/silence_phones.txt
echo LAU >> $locdict/silence_phones.txt
echo NOI >> $locdict/silence_phones.txt

echo SIL > $locdict/optional_silence.txt

# Some downstream scripts expect this file exists, even if empty
touch $locdict/extra_questions.txt

echo "*** Creating phone lists finished!"
