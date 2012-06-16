#!/bin/bash
#
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

# To be run from one directory above this script.

perl -e 'while(<>){ 
    if (m/[WS]ER (\S+)/ && (!defined $bestwer || $bestwer > $1)){ $bestwer = $1; $bestline=$_; } # kaldi "compute-wer" tool.
    elsif (m: (Mean|Sum/Avg|)\s+\|\s+\S+\s+\S+\s+\|\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\|:
        && (!defined $bestwer || $bestwer > $2)){ $bestwer = $2; $bestline=$_; } }  # sclite.
   if (defined $bestline){ print $bestline; } '

