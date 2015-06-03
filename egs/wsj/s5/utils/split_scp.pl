#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
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



# This program splits up any kind of .scp or archive-type file.
# If there is no utt2spk option it will work on any text  file and
# will split it up with an approximately equal number of lines in
# each but.
# With the --utt2spk option it will work on anything that has the
# utterance-id as the first entry on each line; the utt2spk file is
# of the form "utterance speaker" (on each line).
# It splits it into equal size chunks as far as it can.  If you use the utt2spk
# option it will make sure these chunks coincide with speaker boundaries.  In
# this case, if there are more chunks than speakers (and in some other
# circumstances), some of the resulting chunks will be empty and it will print
# an error message and exit with nonzero status.
# You will normally call this like:
# split_scp.pl scp scp.1 scp.2 scp.3 ...
# or
# split_scp.pl --utt2spk=utt2spk scp scp.1 scp.2 scp.3 ...
# Note that you can use this script to split the utt2spk file itself,
# e.g. split_scp.pl --utt2spk=utt2spk utt2spk utt2spk.1 utt2spk.2 ...

# You can also call the scripts like:
# split_scp.pl -j 3 0 scp scp.0
# [note: with this option, it assumes zero-based indexing of the split parts,
# i.e. the second number must be 0 <= n < num-jobs.]

$num_jobs = 0;
$job_id = 0;
$utt2spk_file = "";

for ($x = 1; $x <= 2; $x++) {
    if ($ARGV[0] eq "-j") {
        shift @ARGV;
        $num_jobs = shift @ARGV;
        $job_id = shift @ARGV;
        if ($num_jobs <= 0 || $job_id < 0 || $job_id >= $num_jobs) {
            die "Invalid num-jobs and job-id: $num_jobs and $job_id";
        }
    }
    if ($ARGV[0] =~ "--utt2spk=(.+)") {
        $utt2spk_file=$1;
        shift;
    }
}

if(($num_jobs == 0 && @ARGV < 2) || ($num_jobs > 0 && (@ARGV < 1 || @ARGV > 2))) {
    die "Usage: split_scp.pl [--utt2spk=<utt2spk_file>] in.scp out1.scp out2.scp ... \n" .
        " or: split_scp.pl -j num-jobs job-id [--utt2spk=<utt2spk_file>] in.scp [out.scp]\n" .
        " ... where 0 <= job-id < num-jobs.";
}

$error = 0;
$inscp = shift @ARGV;
if ($num_jobs == 0) { # without -j option
    @OUTPUTS = @ARGV;
} else {
    for ($j = 0; $j < $num_jobs; $j++) {
        if ($j == $job_id) { 
            if (@ARGV > 0) { push @OUTPUTS, $ARGV[0]; }
            else { push @OUTPUTS, "-"; }
        } else {
            push @OUTPUTS, "/dev/null";
        }
    }
}

if ($utt2spk_file ne "") {  # We have the --utt2spk option...
    open(U, "<$utt2spk_file") || die "Failed to open utt2spk file $utt2spk_file";
    while(<U>) {
        @A = split;
        @A == 2 || die "Bad line $_ in utt2spk file $utt2spk_file";
        ($u,$s) = @A;
        $utt2spk{$u} = $s;
    }
    open(I, "<$inscp") || die "Opening input scp file $inscp";
    @spkrs = ();
    while(<I>) {
        @A = split;
        if(@A == 0) { die "Empty or space-only line in scp file $inscp"; }
        $u = $A[0];
        $s = $utt2spk{$u};
        if(!defined $s) { die "No such utterance $u in utt2spk file $utt2spk_file"; }
        if(!defined $spk_count{$s}) {
            push @spkrs, $s; 
            $spk_count{$s} = 0;
            $spk_data{$s} = "";
        }
        $spk_count{$s}++;
        $spk_data{$s} = $spk_data{$s} . $_;
    }
    # Now split as equally as possible ..
    # First allocate spks to files by allocating an approximately
    # equal number of speakers.
    $numspks = @spkrs;  # number of speakers.
    $numscps = @OUTPUTS; # number of output files.
    if ($numspks < $numscps) {
      die "Refusing to split data because number of speakers $numspks is less " .
          "than the number of output .scp files $numscps";
    }
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scparray[$scpidx] = []; # [] is array reference.
    }
    for ($spkidx = 0; $spkidx < $numspks; $spkidx++) {
        $scpidx = int(($spkidx*$numscps) / $numspks);
        $spk = $spkrs[$spkidx];
        push @{$scparray[$scpidx]}, $spk;
        $scpcount[$scpidx] += $spk_count{$spk};
    }

    # Now will try to reassign beginning + ending speakers
    # to different scp's and see if it gets more balanced.
    # Suppose objf we're minimizing is sum_i (num utts in scp[i] - average)^2.
    # We can show that if considering changing just 2 scp's, we minimize
    # this by minimizing the squared difference in sizes.  This is
    # equivalent to minimizing the absolute difference in sizes.  This
    # shows this method is bound to converge.

    $changed = 1;
    while($changed) {
        $changed = 0;
        for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
            # First try to reassign ending spk of this scp.
            if($scpidx < $numscps-1) {
                $sz = @{$scparray[$scpidx]};
                if($sz > 0) {
                    $spk = $scparray[$scpidx]->[$sz-1];
                    $count = $spk_count{$spk};
                    $nutt1 = $scpcount[$scpidx];
                    $nutt2 = $scpcount[$scpidx+1];
                    if( abs( ($nutt2+$count) - ($nutt1-$count))
                        < abs($nutt2 - $nutt1))  { # Would decrease
                        # size-diff by reassigning spk...
                        $scpcount[$scpidx+1] += $count;
                        $scpcount[$scpidx] -= $count;
                        pop @{$scparray[$scpidx]};
                        unshift @{$scparray[$scpidx+1]}, $spk;
                        $changed = 1;
                    }
                }
            }
            if($scpidx > 0 && @{$scparray[$scpidx]} > 0) {
                $spk = $scparray[$scpidx]->[0];
                $count = $spk_count{$spk};
                $nutt1 = $scpcount[$scpidx-1];
                $nutt2 = $scpcount[$scpidx];
                if( abs( ($nutt2-$count) - ($nutt1+$count))
                    < abs($nutt2 - $nutt1))  { # Would decrease
                    # size-diff by reassigning spk...
                    $scpcount[$scpidx-1] += $count;
                    $scpcount[$scpidx] -= $count;
                    shift @{$scparray[$scpidx]};
                    push @{$scparray[$scpidx-1]}, $spk;
                    $changed = 1;
                }
            }
        }
    }
    # Now print out the files...
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scpfn = $OUTPUTS[$scpidx];
        open(F, ">$scpfn") || die "Could not open scp file $scpfn for writing.";
        $count = 0;
        if(@{$scparray[$scpidx]} == 0) {
            print STDERR "Error: split_scp.pl producing empty .scp file $scpfn (too many splits and too few speakers?)\n";
            $error = 1;
        } else {
            foreach $spk ( @{$scparray[$scpidx]} ) {
                print F $spk_data{$spk};
                $count += $spk_count{$spk};
            }
            if($count != $scpcount[$scpidx]) { die "Count mismatch [code error]"; }
        }
        close(F);
    }
} else {
   # This block is the "normal" case where there is no --utt2spk 
   # option and we just break into equal size chunks.

    open(I, "<$inscp") || die "Opening input scp file $inscp";

    $numscps = @OUTPUTS;  # size of array.
    @F = ();
    while(<I>) {
        push @F, $_;
    }
    $numlines = @F;
    if($numlines == 0) {
        print STDERR "split_scp.pl: error: empty input scp file $inscp , ";
        $error = 1;
    }
    $linesperscp = int( $numlines / $numscps); # the "whole part"..
    $linesperscp >= 1 || die "You are splitting into too many pieces!";
    $remainder = $numlines - ($linesperscp * $numscps);
    ($remainder >= 0 && $remainder < $numlines) || die "bad remainder $remainder";
    # [just doing int() rounds down].
    $n = 0;
    for($scpidx = 0; $scpidx < @OUTPUTS; $scpidx++) {
        $scpfile = $OUTPUTS[$scpidx];
        open(O, ">$scpfile") || die "Opening output scp file $scpfile";
        for($k = 0; $k < $linesperscp + ($scpidx < $remainder ? 1 : 0); $k++) {
            print O $F[$n++];
        }
        close(O) || die "Closing scp file $scpfile";
    }
    $n == $numlines || die "split_scp.pl: code error., $n != $numlines";
}

exit ($error ? 1 : 0);
