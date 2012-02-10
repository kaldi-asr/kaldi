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


# This script makes lists of phones that are shared for building the monophone
# system and shared during phone clustering when creating questions for the
# triphone system.  It puts out a line for each "real" phone, and that line
# contains all the versions of the "real" phones.  

# Takes as standard input e.g. data/lang/phones.txt


if [ $# != 0 ]; then
   echo "Usage: make_shared_phones.sh < phones.txt"
   exit 1;
fi

echo "SIL NSN SPN LAU"

# This script reads from the standard input.
grep -v eps | grep -v -E 'SIL|NSN|SPN|LAU' | awk '{print $1}' | \
  perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $position=$3;
     if($phone eq $curphone){ print " $phone$stress$position"; }
  else { if(defined $curphone){ print "\n"; } $curphone=$phone;  print "$phone$stress$position";  }} print "\n"; '

