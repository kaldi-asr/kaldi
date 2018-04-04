#!/usr/bin/env perl
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


# takes in a file list with lines like
# /mnt/matylda2/data/WSJ1/13-16.1/wsj1/si_dt_20/4k0/4k0c030a.wv1
# and outputs an scp in kaldi format with lines like
# 4k0c030a /mnt/matylda2/data/WSJ1/13-16.1/wsj1/si_dt_20/4k0/4k0c030a.wv1
# (the first thing is the utterance-id, which is the same as the basename of the file.


while(<>){
    m:^\S+/(\w+)\.[wW][vV][12]$: || die "Bad line $_";
    $id = $1;
    $id =~ tr/A-Z/a-z/;  # Necessary because of weirdness on disk 13-16.1 (uppercase filenames)
    print "$id $_";
}

