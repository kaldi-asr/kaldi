#!/bin/bash 
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)

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

# begin configuration section.
min_lmwt=7
max_lmwt=17
stage=0
cer=0
ctm_name=
cmd=run.pl
#end configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

set -e
set -o pipefail

data=$1; 
q=$2; 
shift; shift;

if [ -z $ctm_name ] ; then
  ctm_name=`basename $data`;
fi

name=$ctm_name

for i in $@ ; do
    p=$q/`basename $i`
    [ ! -f $i/reco2file_and_channel ] && "The file reco2file_and_channel not present in the $i directory!" && exit 1
    for lmw in $q/score_* ; do
        test -d $lmw || exit 1; #this is to protect us before creating directory "score_*" in cases no real score_[something] directory exists
        d=$p/`basename $lmw`
        mkdir -p $d

        [ ! -f $lmw/$name.ctm ] && echo "File $lmw/$name.ctm does not exist!" && exit 1
        utils/filter_scp.pl <(cut -f 1 -d ' ' $i/reco2file_and_channel) $lmw/$name.ctm > $d/`basename $i`.ctm
    done

    if [ -f $i/stm ] ; then
        local/score_stm.sh --min-lmwt $min_lmwt --max-lmwt $max_lmwt --cer $cer --cmd "$cmd" $i data/lang $p
    else
        echo "Not running scoring, file $i/stm does not exist"
    fi

done
exit 0

