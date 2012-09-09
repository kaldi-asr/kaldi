#!/usr/bin/perl

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


# seeding is optional...
if ($#ARGV==0) {
  srand($ARGV[0]);
} else {
  srand(0); # Seems to give inconsistent behavior if we don't seed.
}


# This script shuffles lines of a list. 
# The list is read from stdin and written to stdout. 
@X = <STDIN>;
@X = sort { rand() <=> rand() } @X;
print @X; 
