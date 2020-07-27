#!/usr/bin/env perl
#
#===============================================================================
# Copyright 2017  Johns Hopkins University (author: Yenda Trmal <jtrmal@gmail.com>)
#                 Johns Hopkins University (author: Daniel Povey)
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
#===============================================================================

# validation script for data/<dataset>/text
# to be called (preferably) from utils/validate_data_dir.sh
use strict;
use warnings;
use utf8;
use Fcntl qw< SEEK_SET >;

# we made a decision a few months back that we only support ASCII or UTF-8.
# By definition, every ASCII (7-bit) is a valid UTF-8.
# I.e. we no longer assume we should be able to process ISO-Latin-1 and other
# 8-bit "ASCII" encodings -- that would make validation impossible, as 
# we do not maintain an internal concept of encoding and broken UTF-8 would
# be seen as a  valid 8-bit ASCII
sub get_unicode_stream {
  use Encode qw(decode encode);
  my $is_utf_compatible = 1;
  my @unicode_lines;
  my @raw_lines;
  my $raw_text;
  my $lineno = 0;
  my $file = shift;

  while (<$file>) {
    $raw_text = $_;
    last unless $raw_text;
    if ($is_utf_compatible) {
      my $decoded_text = eval { decode("UTF-8", $raw_text, Encode::FB_CROAK) } ;
      $is_utf_compatible = $is_utf_compatible && defined($decoded_text);
      push @unicode_lines, $decoded_text;
    } else {
      print STDERR "ERROR: the line number $lineno (containing $raw_text) cannot be interpreted as a valid UTF-8 or (7-bit) ASCII\n";
      return (0, $unicode_lines)
    }
    $lineno += 1;
  }

  return (1, @unicode_lines);
}

# check if the given unicode string contain unicode whitespaces
# other than the usual four: TAB, LF, CR and SPACE
sub validate_utf8_whitespaces {
  my $unicode_lines = shift;
  use feature 'unicode_strings';
  for (my $i = 0; $i < scalar @{$unicode_lines}; $i++) {
    my $current_line = $unicode_lines->[$i];
    if ((substr $current_line, -1) ne "\n"){
      print STDERR "$0: The current line (nr. $i) has invalid newline\n";
      return 1;
    }
    my @A = split(" ", $current_line);
    my $utt_id = $A[0];
    # we replace TAB, LF, CR, and SPACE
    # this is to simplify the test
    if ($current_line =~ /\x{000d}/) {
      print STDERR "$0: The line number $i (key $utt_id) contains CR (0x0D) character (we do not support Windows/DOS-style end-of-line characters)\n";
      return 1;
    }
    $current_line =~ s/[\x{0009}\x{000a}\x{0020}\x{007f}\x{00ff}]/./g;
    if ($current_line =~/\s/) {
      print STDERR "$0: The line number $i (key $utt_id) contains disallowed Unicode whitespaces\n";
      return 1;
    }
  }
  return 0;
}

# checks if the text in the file (supplied as the argument) is utf-8 compatible
# if yes, checks if it contains only allowed whitespaces. If no, then does not
# do anything. The function seeks to the original position in the file after
# reading the text.
sub check_allowed_whitespace {
  my $file = shift;
  my $filename = shift;
  my $pos = tell($file);
  (my $is_utf, my @lines) = get_unicode_stream($file);
  seek($file, $pos, SEEK_SET);
  if ($is_utf) {
    my $has_invalid_whitespaces = validate_utf8_whitespaces(\@lines);
    if ($has_invalid_whitespaces) {
      print STDERR "$0: ERROR: text file '$filename' contains disallowed UTF-8 whitespace character(s)\n";
      return 0;
    }
    return 1;
  }
  return 0;
}

if(@ARGV != 1) {
  die "Usage: validate_text.pl <text-file>\n" .
      "e.g.: validate_text.pl data/train/text\n";
}

my $text = shift @ARGV;

if (-z "$text") {
  print STDERR "$0: ERROR: file '$text' is empty or does not exist\n";
  exit 1;
}

if(!open(FILE, "<$text")) {
  print STDERR "$0: ERROR: failed to open $text\n";
  exit 1;
}

check_allowed_whitespace(\*FILE, $text) or exit 1;
close(FILE);
