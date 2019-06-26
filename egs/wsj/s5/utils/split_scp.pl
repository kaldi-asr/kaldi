#!/usr/bin/env perl

# Copyright 2010-2011 Microsoft Corporation

# See ../../COPYING for clarification regarding multiple authors
#
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

use warnings;

$num_jobs = 0;
$job_id = 0;
$utt2spk_file = "";
$one_based = 0;

for ($x = 1; $x <= 3 && @ARGV > 0; $x++) {
    if ($ARGV[0] eq "-j") {
        shift @ARGV;
        $num_jobs = shift @ARGV;
        $job_id = shift @ARGV;
    }
    if ($ARGV[0] =~ /--utt2spk=(.+)/) {
        $utt2spk_file=$1;
        shift;
    }
    if ($ARGV[0] eq '--one-based') {
        $one_based = 1;
        shift @ARGV;
    }
}

if ($num_jobs != 0 && ($num_jobs < 0 || $job_id - $one_based < 0 ||
                       $job_id - $one_based >= $num_jobs)) {
  die "$0: Invalid job number/index values for '-j $num_jobs $job_id" .
      ($one_based ? " --one-based" : "") . "'\n"
}

$one_based
    and $job_id--;

if(($num_jobs == 0 && @ARGV < 2) || ($num_jobs > 0 && (@ARGV < 1 || @ARGV > 2))) {
    die
"Usage: split_scp.pl [--utt2spk=<utt2spk_file>] in.scp out1.scp out2.scp ...
   or: split_scp.pl -j num-jobs job-id [--one-based] [--utt2spk=<utt2spk_file>] in.scp [out.scp]
 ... where 0 <= job-id < num-jobs, or 1 <= job-id <- num-jobs if --one-based.\n";
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
    open($u_fh, '<', $utt2spk_file) || die "$0: Error opening utt2spk file $utt2spk_file: $!\n";
    while(<$u_fh>) {
        @A = split;
        @A == 2 || die "$0: Bad line $_ in utt2spk file $utt2spk_file\n";
        ($u,$s) = @A;
        $utt2spk{$u} = $s;
    }
    close $u_fh;
    open($i_fh, '<', $inscp) || die "$0: Error opening input scp file $inscp: $!\n";
    @spkrs = ();
    while(<$i_fh>) {
        @A = split;
        if(@A == 0) { die "$0: Empty or space-only line in scp file $inscp\n"; }
        $u = $A[0];
        $s = $utt2spk{$u};
        if(!defined $s) { die "No such utterance $u in utt2spk file $utt2spk_file"; }
        if(!defined $spk_data{$s}) {
            push @spkrs, $s;
            $spk_data{$s} = [];  # ref to new empty array.
        }
        $scpcount++;
        push @{$spk_data{$s}}, $_;
    }
    # Now split as equally as possible ..
    # First allocate spks to files by allocating an approximately
    # equal number of speakers.
    $numspks = @spkrs;  # number of speakers.
    $numscps = @OUTPUTS; # number of output files.
    if ($scpcount < $numscps) {
      die "Refusing to split data because number of utt $numspks is less " .
          "than the number of output .scp files $numscps";
    }
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scparray[$scpidx] = []; # [] is array reference.
    }

    # Now will try to reassign beginning + ending utts
    # to different split* and see if it gets more balanced.

    $loopDone = 0;
    $numLoop = 0;
    until($loopDone){
        $doneCount = 0;
        foreach  $spk (@spkrs) {
            $line = pop @{$spk_data{$spk}};
            if(!defined $line){
                $doneCount += 1;
                if($numspks == $doneCount){
                    $loopDone = 1;
                }
            }else{
                $scpidx = $numLoop % $numscps;
                push @{$scparray[$scpidx]}, $line;
            }
        }
        $numLoop += 1;
    }
    # Now print out the files...
    $count = 0;
    for($scpidx = 0; $scpidx < $numscps; $scpidx++) {
        $scpfile = $OUTPUTS[$scpidx];
        ($scpfile ne '-' ? open($f_fh, '>', $scpfile)
                         : open($f_fh, '>&', \*STDOUT)) ||
            die "$0: Could not open scp file $scpfile for writing: $!\n";
        
        if(@{$scparray[$scpidx]} == 0) {
            print STDERR "$0: eError: split_scp.pl producing empty .scp file " .
                         "$scpfile (too many splits and too few speakers?)\n";
            $error = 1;
        } else {
            print $f_fh @{$scparray[$scpidx]};
        }
        $count += @{$scparray[$scpidx]};
        close($f_fh);
    }
    if($count != $scpcount) { die "Count mismatch [code error]"; }
} else {
   # This block is the "normal" case where there is no --utt2spk
   # option and we just break into equal size chunks.

    open($i_fh, '<', $inscp) || die "$0: Error opening input scp file $inscp: $!\n";

    $numscps = @OUTPUTS;  # size of array.
    @F = ();
    while(<$i_fh>) {
        push @F, $_;
    }
    $numlines = @F;
    if($numlines == 0) {
        print STDERR "$0: error: empty input scp file $inscp\n";
        $error = 1;
    }
    $linesperscp = int( $numlines / $numscps); # the "whole part"..
    $linesperscp >= 1 || die "$0: You are splitting into too many pieces! [reduce \$nj]\n";
    $remainder = $numlines - ($linesperscp * $numscps);
    ($remainder >= 0 && $remainder < $numlines) || die "bad remainder $remainder";
    # [just doing int() rounds down].
    $n = 0;
    for($scpidx = 0; $scpidx < @OUTPUTS; $scpidx++) {
        $scpfile = $OUTPUTS[$scpidx];
        ($scpfile ne '-' ? open($o_fh, '>', $scpfile)
                         : open($o_fh, '>&', \*STDOUT)) ||
            die "$0: Could not open scp file $scpfile for writing: $!\n";
        for($k = 0; $k < $linesperscp + ($scpidx < $remainder ? 1 : 0); $k++) {
            print $o_fh $F[$n++];
        }
        close($o_fh) || die "$0: Eror closing scp file $scpfile: $!\n";
    }
    $n == $numlines || die "$n != $numlines [code error]";
}

exit ($error);