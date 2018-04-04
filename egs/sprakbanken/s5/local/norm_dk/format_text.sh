#!/bin/bash

# Copyright 2014 Andreas Kirkedal

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

#dir=norm_dk

#dos2unix $2

mode=$1

tmp="$(mktemp -d /tmp/kaldi.XXXX)"

dir=$(pwd)/local/norm_dk

src=$tmp/src.tmp
abbr=$tmp/anot.tmp
rem=$tmp/rem.tmp
line=$tmp/line.tmp
num=$tmp/num.tmp
nonum=$tmp/nonum.tmp

cat $2 | tr -d '\r' > $src

#$dir/expand_abbr_medical.sh $src > $abbr;
$dir/remove_annotation.sh $src > $rem;
if [ $mode != "am" ]; then
    $dir/sent_split.sh $rem > $line;
else
    $dir/write_out_formatting.sh $rem > $line;
fi

$dir/expand_dates.sh $line |\
$dir/format_punct.sh  >  $num;
#python3 $dir/writenumbers.py $dir/numbersUp.tbl $num $nonum;
# $dir/write_punct.sh | \
cat $num | \
perl -pi -e "s/^\n//" | \
perl -pe 's/ (.{4}.*?)\./ \1/g'
# | PERLIO=:utf8 perl -pe '$_=lc'

# Comment this line for debugging
wait
rm -rf $tmp
