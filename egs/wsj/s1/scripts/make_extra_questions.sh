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


# This is specialized for this particular WSJ setup.
# It creates the extra questions that specialize within each phone to ask
# about the stress markers and positions.  We create one question for each stress-marker
# type and each position.
# We also make a questions about silences.

# Takes as standard input data/phones.txt

# Question about silence category of phones.
echo "SIL NSN SPN"; 

cat data/phones.txt | grep -v eps | awk '{print $1}' | \
  perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $pos=$3;
     $full_phone ="$1$2$3";
     $pos2list{$pos} = $pos2list{$pos} .  $full_phone . " ";
     $stress2list{$stress} = $stress2list{$stress} .  $full_phone . " ";
   } 
   foreach $k (keys %pos2list) { print "$pos2list{$k}\n"; } 
   foreach $k (keys %stress2list) { print "$stress2list{$k}\n"; }  '


