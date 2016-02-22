#!/usr/bin/env perl


# This program reads and writes either a dictionary or just a list
# of words, and it removes any words containing ";" or "," as these
# are used in these programs.  It will warn about these.
# It will die if the pronunciations have these symbols in.
while(<>) {
  chop;
  @A = split(" ", $_);
  $word = shift @A;
  
  if ($word =~ m:[;,]:) {
    print STDERR "Omitting line $_ since it has one of the banned characters ; or ,\n" ;
  } else {
    $_ =~ m:[;,]: && die "Phones cannot have ; or , in them.";
    print $_ . "\n";
  }
}
