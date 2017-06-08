#!/usr/bin/perl
# Copyright 2015 GoVivace Inc. (Author: Nagendra Kumar Goel)

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

# Analyze a segments file and print important stats on it.

$dur = $total = 0;
$maxDur = 0;
$minDur = 9999999999;
$n = 0;
while(<>){
    chomp;
    @t = split(/\s+/);
    $dur = $t[3] - $t[2];
    $total += $dur;
    if ($dur > $maxDur) {
        $maxSegId = $t[0];
        $maxDur = $dur;
    }
    if ($dur < $minDur) {
        $minSegId = $t[0];
        $minDur = $dur;
    }
    $n++;
}
$avg=$total/$n;
$hrs = $total/3600;
print "Total $hrs hours of data\n";
print "Average segment length $avg seconds\n";
print "Segment $maxSegId has length of $maxDur seconds\n";
print "Segment $minSegId has length of $minDur seconds\n";
