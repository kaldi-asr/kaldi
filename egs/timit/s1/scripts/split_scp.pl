#!/usr/bin/perl -w
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
# It splits it into equal size chunks as far as it can.  If you use
# the utt2spk option it will make sure these chunks coincide with
# speaker boundaries.  In this case, if there are more chunks
# than speakers (and in some other circumstances), some of the 
# resulting  chunks will be empty and it
# will print a warning.
# You will normally call this like:
# split_scp.pl scp scp.1 scp.2 scp.3 ...
# or
# split_scp.pl --utt2spk=utt2spk scp scp.1 scp.2 scp.3 ...
# Note that you can use this script to split the utt2spk file itself,
# e.g. split_scp.pl --utt2spk=utt2spk utt2spk utt2spk.1 utt2spk.2 ...

if(@ARGV < 2 ) {
    die "Usage: split_scp.pl [--utt2spk=<utt2spk_file>] in.scp out1.scp out2.scp ... ";
}

if($ARGV[0] =~ m:^-:) {  
    # Everything inside this block
    # corresponds to what we do when the --utt2spk option is used.
    $opt = shift @ARGV;
    @A = split("=", $opt);
    if(@A != 2 || $A[0] ne "--utt2spk") {
        die "split_scp.pl: invalid option $ARGV[0]";
    }
    $utt2spk_file = $A[1];
    open(U, "<$utt2spk_file") || die "Failed to open utt2spk file $utt2spk_file";
    while(<U>) {
        @A = split;
        @A == 2 || die "Bad line $_ in utt2spk file $utt2spk_file";
        ($u,$s) = @A;
        $utt2spk{$u} = $s;
    }
    $inscp = shift @ARGV;
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
    # First allocate spks to files by given approximately
    # equal #spks.
    $numspks = @spkrs;  # number of speakers.
    $numscps = @ARGV; # number of output files.
    $spksperscp = int( ($numspks+($numscps-1)) / $numscps); # the +$(numscps-1) forces rounding up.
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scparray[$scpidx] = []; # [] is array reference.
        for($n = $spksperscp * $scpidx; 
            $n < $numspks && $n < $spksperscp*($scpidx+1); 
            $n++) {
            $spk = $spkrs[$n];
            push @{$scparray[$scpidx]}, $spk;
            $scpcount[$scpidx] += $spk_count{$spk};
        }
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
        $scpfn = $ARGV[$scpidx];
        open(F, ">$scpfn") || die "Could not open scp file $scpfn for writing.";
        $count = 0;
        if(@{$scparray[$scpidx]} == 0) {
            print STDERR "Warning: split_scp.pl producing empty .scp file $scpfn (too many splits and too few speakers?)";
        }
        foreach $spk ( @{$scparray[$scpidx]} ) {
            print F $spk_data{$spk};
            $count += $spk_count{$spk};
        }
        if($count != $scpcount[$scpidx]) { die "Count mismatch [code error]"; }
        close(F);
    }
} else { 
   # This block is the "normal" case where there is no --utt2spk 
   # option and we just break into equal size chunks.

    $inscp = shift @ARGV;
    open(I, "<$inscp") || die "Opening input scp file $inscp";

    $numscps = @ARGV;  # size of array.
    @F = ();
    while(<I>) {
        push @F, $_;
    }
    $numlines = @F;
    if($numlines == 0) {
        print STDERR "split_scp.pl: warning: empty input scp file $inscp";
    }
    $linesperscp = int( ($numlines+($numscps-1)) / $numscps); # the +$(numscps-1) forces rounding up.
# [just doing int() rounds down].
    for($scpidx = 0; $scpidx < @ARGV; $scpidx++) {
        $scpfile = $ARGV[$scpidx];
        open(O, ">$scpfile") || die "Opening output scp file $scpfile";
        for($n = $linesperscp * $scpidx; $n < $numlines && $n < $linesperscp*($scpidx+1); $n++) {
            print O $F[$n];
        }
        close(O) || die "Closing scp file $scpfile";
    }
}
