#!/usr/bin/perl
# Copyright 2012 Navdeep Jaitly.
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



# usage:  make_trans.sh prefix in.flist out.txt out.scp

# prefix is first letters of the database "key" (rest are numeric)

# in.flist is just a list of the WAV file paths (X.WAV). The 
# monophone transcriptions are in the files (X.phn).
# out.txt is the output transcriptions in format "key word1 word\n"
# out.scp is the output scp file, which is as in.scp but has the
# database-key first on each line.

# Reads from first argument in.flist
# Writes to standard output trans.txt

sub ParseTranscript() {
    my $transcript_file = $_[0];
    open(F, "<$transcript_file") || die "Error opening phone transcription file $transcript_file\n";
    my $trans = "h#" ; 
    my $line = <F> ; 
    chomp ($line);
    # first line should be "h#".
    ($line =~/h#/) || die "First line should be h#. Got line: $line";
    my @pieces;
    while(<F>) {
        chomp ; 
        @pieces = split(" ", $_);
        @pieces == 3 || die "Error parsing file: $transcript_file, line: $_. Expected 3 fields. Found @pieces";
        $trans = $trans . " " . $pieces[2];
    }
    ($pieces[2] =~/^h#/) || die "Last line should be h#";
    #$trans =~s/^h#/<s>/ ; # first h#
    #$trans =~s/h#/<\\s>/ ; # last h#
    $trans =~s/^h#// ; # first h#
    $trans =~s/h#$// ; # last h#
    ($trans !~ m/h#/) || die "Found h# character in transcript, other than start or end.";
    
    close(F);
    return $trans ; 
}

if(@ARGV != 4) {
    die "usage:  make_trans.sh prefix in.flist out.txt out.scp\n";
}
($prefix, $in_flist, $out_txt, $out_scp) = @ARGV;

open(G, "<$in_flist") || die "Opening file list $in_flist";

open(O, ">$out_txt") || die "Open output transcription file $out_txt";

open(P, ">$out_scp") || die "Open output scp file $out_scp";

while(<G>) {
    my $sph_file = $_ ;
    chomp ($sph_file) ;
    $_ =~ m:/(\w+)/(\w+)\.WAV\s+$:i || die "bad scp line $_";
    $spkname = $1;
    $uttname = $2;
    $uttname  =~ tr/a-z/A-Z/;
    $spkname =~ s/_//g; # remove underscore from spk name to make key nicer.
    $key = $prefix . "_" . $spkname . "_" . $uttname;
    $key =~ tr/A-Z/a-z/; # Make it all lower case.
     # to make the numerical and string-sorted orders the same.
    my $transcript_file = substr($_, 0, length($_)-4) . "phn";
    if (! -e $transcript_file ) {
       $transcript_file = substr($_, 0, length($_)-4) . "PHN";
    }
    if (! -e $transcript_file ) {
       print "Transcription file: $transcript_file missing." ; 
    }
     
    my $trans = &ParseTranscript($transcript_file);
    $trans =~ tr/a-z/A-Z/; # Make it all upper case.
    print P "$key $sph_file\n";
    print O "$key $trans\n";
    $n++;
} 
close(O) || die "Closing output.";
close(P) || die "Closing output.";


