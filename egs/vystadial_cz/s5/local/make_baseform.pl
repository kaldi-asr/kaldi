#!perl -w

#
# ====================================================================
# Copyright (C) 1999-2008 Carnegie Mellon University and Alexander
# Rudnicky. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# This work was supported in part by funding from the Defense Advanced
# Research Projects Agency, the Office of Naval Research and the National
# Science Foundation of the United States of America, and by member
# companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
# the contributions of many volunteers to the expansion and improvement of
# this dictionary.
#
# THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
# NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ====================================================================
#

# [20050309] (air) Created.
# strip out stress marks from a cmudict, producing a "SphinxPhones_40" dictionary
# [20080420] (air) Changed to pass comments.
#                  Fixed output collation sequence; DOS eol's
# [20090309] (air) fixed duplicate pron and collation bugs
# [20090331] (air) restored standard collation order (since other stuff deppends on it)
# [20090629] (air) do not put comments into SPHINX_40 version; not all software deals
# [20100118] (air) added $VERBOSE; this should really be a cmdline flag...
#


$VERBOSE = 0;

my $basecount = 0;
my $dupl = 0;
my $base = 0;
my $varia = 0;

if ( scalar @ARGV ne 2 ) { die "usage: make_baseform <input> <output>\n"; }

open(IN, $ARGV[0]) || die "can't open $ARGV[0] for reading!\n";
open(OUT,">$ARGV[1]") || die "can't open $ARGV[1] for writing!\n";

@header = ();  # header comment lines (passed through)
%dict = ();    # words end up in here
%histo = ();   # some statistics on variants

get_dict(\%dict,\@header,IN);  # process the entries

# what have we got?
print STDERR "$basecount forms processed\n";
print STDERR "$base baseforms, $varia variants and $dupl duplicates found.\n";
print STDERR "variant distribution:\n";
foreach $var ( sort keys %histo ) {
    print STDERR "$var\t$histo{$var}\n";
}

# print special comments (copyright, etc.)
# removed since it messes some things up...
# foreach $h (@header) { print OUT "$h\n"; }

# print out each entry
%dict_out = ();
foreach $w (sort keys %dict) {
  $var=1;  # variants will number starting with 2
  foreach $p ( @{$dict{$w}} ) {
    if ($var eq 1) {
	$dict_out{$w} = $p;
      $var++;
    }  else {
      $dict_out{"$w($var)"} = $p;
      $var++;
    }
  }
}

foreach $entry ( sort keys %dict_out ) {
    print OUT "$entry\t$dict_out{$entry}\n";
}

close(IN);
close(OUT);

#
#
# read in a dictionary
sub get_dict {
  my $dict = shift;  # data structure with dictionary entries
  my $header = shift;
  my $target = shift;  # input file handle

  while (<$target>) {
    s/[\r\n]+$//g;  # DOS-robust chomp;

    # process comments; blank lines ignored
    # presume that ";;; #" will be collected and emitted at the top
    if ($_ =~ /^;;; \#/) { push @$header, $_; next; }  # save header info
    elsif ( $_ =~ /^;;;/ ) { next; }  # ignore plain comments
    elsif ( $_ =~ /^\s*$/ ) { next; }  # ignore blank lines

    # extract the word,pron pair and prepare for processing
    ($word,$pron) = /(.+?)\s+(.+?)$/;
    if (! defined $word) { print STDERR "bad entry (no head word): $_\n"; next; }

    $basecount++;

    if ($word =~ /\)$/) { # variant
      ($root,$variant) = ($word =~ m/(.+?)\((.+?)\)/);
    } else {
      $root = $word;
      $variant = 0;
    }
    $pron = &strip_stress($pron);

    # found a new baseform; set it up
    if ( ! defined $dict->{$root} ) {
	$dict->{$root}[0] = $pron;
	$base++;
	next;
    }

    # old baseform; see if, after removed stress, pron is a duplicate
    foreach $var ( @{$dict->{$root}} ) {
	if ( $var eq $pron ) {
	    if ($VERBOSE) {print STDERR "duplicate entry: $root ($variant) $pron\n";}
	    $dupl++;
	    $pron = "";
	    last;
	}
    }

    # it's a new variant on an existing baseform, keep it
    if ( $pron ne "" ) { 
	push @{$dict->{$root}}, $pron;
	$varia++;
	$histo{scalar @{$dict->{$root}}}++;  # track variant stats
	if ( scalar @{$dict->{$root}} > 4 ) { print STDERR "$root -- ",scalar @{$dict->{$root}},"\n"; }
    }
  }
}


# strip stress marks from phonetic symbols
sub strip_stress {
  @pron = split " ", $_[0];
  my $p;
  foreach $p (@pron) { if ( $p =~ /\d$/) { $p =~ s/(\d+)$//; } }
  return ( join(" ",@pron));
}

#
