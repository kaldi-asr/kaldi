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


# This script is part of a diagnostic step when using exponential transforms.

$map=$ARGV[0]; open(M,"<$map")||die "opening map file $map";
while(<M>){ @A=split(" ",$_); $map{$A[0]} = $A[1]; }
while(<STDIN>){  
    ($spk,$warp)=split(" ",$_); 
    $class = int($class/2);
    defined $map{$spk} || die "No gender info for speaker $spk";
    $warps{$map{$spk}} = $warps{$map{$spk}} . "$warp ";
}
@K = sort keys %warps;
@K==2||die "wrong number of keys [empty warps file?]";
foreach $k ( @K ) {
    $s =  join(" ", sort { $a <=> $b } ( split(" ", $warps{$k}) )) ;
    print "$k = [ $s ];\n";
} 
# f,m may be reversed below; doesnt matter.
foreach $w ( split(" ", $warps{$K[0]}) ) {
    $nf += 1; $sumf += $w; $sumf2 += $w*$w;
}
foreach $w ( split(" ", $warps{$K[1]}) ) {
    $nm += 1; $summ += $w; $summ2 += $w*$w;
}
$sumf /= $nf; $sumf2 /= $nf;
$summ /= $nm; $summ2 /= $nm;
$sumf2 -= $sumf*$sumf;
$summ2 -= $summ*$summ;
$avgwithin = 0.5*($sumf2+$summ2 );
$diff = abs($sumf - $summ) / sqrt($avgwithin);
print "% class separation is $diff\n"; 
