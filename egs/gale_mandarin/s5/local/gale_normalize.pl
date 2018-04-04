#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright Chao Weng 

# normalizations for hkust trascript
# see the docs/trans-guidelines.pdf for details

while (<STDIN>) {
  @A = split(" ", $_);
  for ($n = 0; $n < @A; $n++) { 
    $a = $A[$n];
    if (($a eq "{breath}")||($a eq "{cough}")||($a eq "{sneeze}")
       || ($a eq "{lipsmack}")) {print "[VOCALIZED-NOISE] "; next;}
    if (($a eq "{laugh}")) {print "[LAUGHTER] "; next;}
    if (($a eq "<noise>")) {print "[NOISE] "; next;}
    $tmp = $a;
    if ($tmp =~ /[^.,?+-]{0,}[.,?+-]+/) { $tmp =~ s:([^.,?+-]{0,})[.,?+-]+::; }
    if ($tmp =~ /。/) { $tmp =~ s:。::g; }
    $tmp =~ s:Ａ:A:g;
    $tmp =~ s:Ｄ:D:g;
    $tmp =~ s:Ｎ:D:g;
    $tmp =~ s:Ⅱ::g;
    $tmp =~ s:　::g;
    $tmp =~ s:、::g;
    $tmp =~ s:】::g;

    if ($tmp =~ /？/) { $tmp =~ s:？::g; }
    if ($tmp =~ /！/) { $tmp =~ s:！::g; }
    if ($tmp =~ /，/) { $tmp =~ s:，::g; }
    if ($tmp =~ /\~[A-Z]/) { $tmp =~ s:\~([A-Z])::; }
    if ($tmp =~ /%\S/) { $tmp =~ s:%(\S)::; }
    if ($tmp =~ /[a-zA-Z]/) {$tmp=uc($tmp);} 
    print "$tmp "; 
  }
  print "\n"; 
}
