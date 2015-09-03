#!/usr/bin/env perl
# Copyright 2010-2011 Microsoft Corporation
#                2015 Guoguo Chen

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

# This script replaces epsilon with #0 on the input side only, of the G.fst
# acceptor.  

while(<>){
  if (/\s+#0\s+/) {
    print STDERR "$0: ERROR: LM has word #0, " .
                 "which is reserved as disambiguation symbol\n";
    exit 1;
  }
  s:^(\d+\s+\d+\s+)\<eps\>(\s+):$1#0$2:;
  print;
}
