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


# This script changes the format of a transcript or reference file from the
# Kaldi text format like:
# utterance-id word1 word2 ....
# to the trn style of format like:
# word1 word2 ... (utterance-id)
# which seems to be what sclite expects in hyp and ref files.

while(<>) {
    @A = split(" ", $_);
    $id = shift @A;
    print join(" ", @A) . "($id)" . "\n";
}
