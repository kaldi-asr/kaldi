#!/usr/bin/perl
# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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


# usage:  make_trans.sh postfix in.flist input.snr out.txt out.scp

# postfix is last letters of the database "key" (rest are numeric)

# in.flist is just a list of filenames, probably of .sph files.
# input.snr is an snr format file from the RM dataset.  
# out.txt is the output transcriptions in format "key word1 word\n"
# out.scp is the output scp file, which is as in.scp but has the
# database-key first on each line.

# Reads from first argument e.g. $rootdir/rm1_audio1/rm1/doc/al_sents.snr
# and second argument train_wav.scp 
# Writes to standard output trans.txt

if(@ARGV != 5) {
    die "usage:  make_trans.sh postfix in.flist input.snr out.txt out.scp\n";
}
($postfix, $in_flist, $input_snr, $out_txt, $out_scp) = @ARGV;

open(F, "<$input_snr") || die "Opening SNOR file $input_snr";

while(<F>) {
    if(m/^;/) { next; }
    m/(.+) \((.+)\)/ || die "bad line $_";
    $T{$2} = $1;
}

close(F);
open(G, "<$in_flist") || die "Opening file list $in_flist";

open(O, ">$out_txt") || die "Open output transcription file $out_txt";

open(P, ">$out_scp") || die "Open output scp file $out_scp";

while(<G>) {
    $_ =~ m:/(\w+)/(\w+)\.sph\s+$:i || die "bad scp line $_";
    $spkname = $1;
    $uttname = $2;
    $uttname  =~ tr/a-z/A-Z/;
    defined $T{$uttname} || die "no trans for sent $uttname";
    $spkname =~ s/_//g; # remove underscore from spk name to make key nicer.
    $key = $spkname . "_" . $uttname . "_" . $postfix;
    $key =~ tr/A-Z/a-z/; # Make it all lower case.
    # to make the numerical and string-sorted orders the same.
    print O "$key $T{$uttname}\n";
    print P "$key $_" || die "Error writing to sph file list";
    $n++;
}
close(O) || die "Closing output.";
close(P) || die "Closing output.";
