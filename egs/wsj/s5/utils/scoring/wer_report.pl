#!/usr/bin/env perl
# Copyright 2015 Johns Hopkins University (author: Jan Trmal <jtrmal@gmail.com>)

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


# This script reads per-utt table generated for example during scoring
# and outpus the WER similar to the format the compute-wer utility 
# or the utils/best_wer.pl produces
# i.e. from table containing lines in this format
# SUM raw 23344 243230 176178 46771 9975 20281 77027 16463
# produces something output like this
# %WER 31.67 [ 77027 / 243230, 9975 ins, 20281 del, 46771 sub ] 
# NB: if the STDIN stream will contain more of the SUM raw entries,
#     the best one will be found and printed 
#
# If the script is called with parameters, it uses them pro provide 
# a description of the output
# i.e.
# cat per-spk-report | utils/scoring/wer_report.pl Full set
# the following output will be produced
# %WER 31.67 [ 77027 / 243230, 9975 ins, 20281 del, 46771 sub ] Full set


while (<STDIN>) {
  if ( m:SUM\s+raw:) {
    @F = split;
    if ((!defined $wer) || ($wer > $F[8])) {
      $corr=$F[4];
      $sub=$F[5];
      $ins=$F[6];
      $del=$F[7];
      $wer=$F[8];
      $words=$F[3];
    }
  }
}

if (defined $wer) {
  $wer_str = sprintf("%.2f", (100.0 * $wer) / $words);
  print "%WER $wer_str [ $wer / $words,  $ins ins, $del del, $sub sub ]";
  print " " . join(" ", @ARGV) if @ARGV > 0;
  print "\n";
}
