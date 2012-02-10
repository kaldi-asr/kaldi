#!/bin/bash
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


# This is specialized for WSJ.
# Making roots that are shared within each phone except we also share the different
# stress-markers for each phone.  Silence and noise phones have a single shared (tied)
# state and are not split.

# Takes as standard input data/phones.txt

echo 'not-shared not-split SIL SPN NSN'

cat data/phones.txt | grep -v eps | grep -v SIL | grep -v SPN | grep -v NSN | awk '{print $1}' | \
  perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
           $phone=$1; $stress=$2; $position=$3;
     if($phone eq $curphone){ print " $phone$stress$position"; }
     else { if(defined $curphone){ print "\n"; } $curphone=$phone; 
           print "shared split $phone$stress$position";  }} print "\n"; '

