#!/usr/bin/perl -w
# Copyright Chao Weng 

# normalizations for hkust trascript
# see the docs/trans-guidelines.pdf for details

while (<STDIN>) {
  @A = split(" ", $_);
  print "$A[0] ";
  for ($n = 1; $n < @A; $n++) { 
    $a = $A[$n];
    if (($a eq "{breath}")||($a eq "{cough}")||($a eq "{sneeze}")
       || ($a eq "{lipsmack}")) {print "[VOCALIZED-NOISE] "; next;}
    if (($a eq "{laugh}")) {print "[LAUGHTER] "; next;}
    if (($a eq "<noise>")) {print "[NOISE] "; next;}
    $tmp = $a;
    if ($tmp =~ /[^.,?+-]{0,}[.,?+-]+/) { $tmp =~ s:([^.,?+-]{0,})[.,?+-]+:$1:; }
    if ($tmp =~ /\~[A-Z]/) { $tmp =~ s:\~([A-Z]):$1:; }
    if ($tmp =~ /%\S/) { $tmp =~ s:%(\S):$1:; }
    if ($tmp =~ /[a-zA-Z]/) {$tmp=uc($tmp);} 
    print "$tmp "; 
  }
  print "\n"; 
}
