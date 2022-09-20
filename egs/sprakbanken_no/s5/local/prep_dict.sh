

#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)
# Copyright 2022  Institute of Language and Speech Processing (ILSP), AthenaRC (Author: Thodoris Kouzelis)

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

local_dir=data/local/dict
lexicon_raw=$local_dir/lexicon.txt


mkdir -p data/local/dict

# Download NoFA dictionary.
wget https://www.nb.no/sbfil/verktoy/nofa/NoFA-1_0.tar.gz -P data/local/data/download/
tar -xzf data/local/data/download/NoFA-1_0.tar.gz -C $local_dir
cp $local_dir/NoFA/lexicon/NoFA-lex.dict $lexicon_raw
rm -rf data/local/lang/NoFA/


# non-silence
cut -d ' ' -f 2- $lexicon_raw | sed 's/ /\n/g' |  sort -u | sed '/SPN/d' > $local_dir/nonsilence_phones.txt
# silence_phones
printf 'SIL\nSPN\n' > $local_dir/silence_phones.txt

# optional_silence
echo 'SIL' > $local_dir/optional_silence.txt

echo '<UNK> SPN' >> $lexicon_raw


