#!/usr/bin/perl

# Copyright 2013 MERL (author: Felix Weninger)

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

use strict;

my $opt_lmw;
my $lm = "bg_5k";

while ($#ARGV > -1) {
    if ($ARGV[0] =~ /^--lmw=(\d+)$/)
    {
        $opt_lmw = $1 + 0;
        shift @ARGV;
    }
    elsif ($ARGV[0] =~ /^--lm=(\w+)$/) {
        $lm = $1;
        shift @ARGV;
    }
    else {
        last;
    }
}


print "$0 @ARGV\n";

my $system = "tri2b_mc";
if ($ARGV[0] ne "") { $system = $ARGV[0]; }

for my $dt_or_et ("dt", "et") {

print "#### RESULTS FOR $dt_or_et ##### \n\n";

my $pref = "REVERB_$dt_or_et";
#if ($lm ne "bg_5k") {
$pref = "${lm}_$pref";
#}
if ($ARGV[1] ne "") { $pref = $ARGV[1] . '_' . $pref; }
if ($ARGV[2] ne "") { $pref = $pref . '_' . $ARGV[2]; }

my $suff = "";

print "exp/$system/decode_$suff$pref*\n";
my @folders = glob("exp/$system/decode_$suff$pref*");

my ($min_lmw, $max_lmw) = (9, 20);
@folders = grep { -f "$_/wer_$min_lmw" } @folders;
my @sum_wer;
my %wer;
my %avg_wer_disp;
my $nc = 0;
my $ns = 0;
my $nr = 0;
for my $lmw ($min_lmw..$max_lmw)
{
    for my $fold (@folders) {
        my $res_file = "$fold/wer_$lmw";
        #print "fold = $fold pref = $pref\n";
        #my ($cond) = $fold =~ /decode_(\w+)$/;
        my ($cond) = $fold =~ /decode_\Q$suff\E\Q${pref}\E_(\w+)$/;
        if ($cond =~ /^Sim.+(far|near|cln)|^Real/) {
            open(RES, $res_file) or die "$res_file: $_";
            while (<RES>) {
                if (/%WER\s+(\S+)/) {
                    my $wer = $1;
                    #print "cond = $cond lmw = $lmw wer = $1\n";
                    if ($cond !~ /cln/) {
                        $sum_wer[$lmw] += $wer;
                    }
                    $wer{$cond}[$lmw] = $wer;
                }
            }
            #print "cond = $cond fold = $fold\n";
        }
    }   
}

if (!$opt_lmw && $dt_or_et eq "dt") {
    $opt_lmw = $min_lmw;
    for my $lmw ($min_lmw+1..$max_lmw) {
        if ($sum_wer[$lmw] < $sum_wer[$opt_lmw]) {
            $opt_lmw = $lmw;
        }
    }
}

print "LMW = $opt_lmw\n";
for my $cond (sort keys %wer) {
    print "$cond\t$wer{$cond}[$opt_lmw]\n";
    if ($cond =~ /SimData_[de]t/) {
        if ($cond !~ /cln/) {
            $avg_wer_disp{"SimData"} += ($wer{$cond}[$opt_lmw] - $avg_wer_disp{"SimData"}) / ++$ns;
        }
        else {
            $avg_wer_disp{"CleanData"} += ($wer{$cond}[$opt_lmw] - $avg_wer_disp{"CleanData"}) / ++$nc;
        }
    }
    elsif ($cond =~ /RealData_[de]t/) {
        $avg_wer_disp{"RealData"} += ($wer{$cond}[$opt_lmw] - $avg_wer_disp{"RealData"}) / ++$nr;
    }
}

#print "Avg_Clean($nc)\t", sprintf("%.2f", $avg_wer_disp{"CleanData"}), "\n";
print "Avg_Sim($ns)\t", sprintf("%.2f", $avg_wer_disp{"SimData"}), "\n";
print "Avg_Real($nr)\t", sprintf("%.2f", $avg_wer_disp{"RealData"}), "\n";
print "\n\n";

}
