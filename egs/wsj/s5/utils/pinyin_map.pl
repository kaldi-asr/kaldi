#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

$num_args = $#ARGV + 1;
if ($num_args != 1) {
  print "\nUsage: pinyin2phone.pl pinyin2phone\n";
  exit;
}

open(MAPS, $ARGV[0]) or die("Could not open pinyin map file.");
my %py2ph; foreach $line (<MAPS>) { @A = split(" ", $line);
  $py = shift(@A);
  $py2ph{$py} = [@A]; 
}

#foreach $word ( keys %py2ph ) {
     #foreach $i ( 0 .. $#{ $py2ph{$word} } ) {
     #    print " $word = $py2ph{$word}[$i]";
     #}
     #print " $#{ $py2ph{$word} }";
     #print "\n";
#}

my @entry;

while (<STDIN>) {
  @A = split(" ", $_);
  @entry = (); 
  $W = shift(@A);
  push(@entry, $W);
  for($i = 0; $i < @A; $i++) {
    $initial= $A[$i]; $final = $A[$i];
    #print $initial, " ", $final, "\n";
    if ($A[$i] =~ /^CH[A-Z0-9]+$/) {$initial =~ s:(CH)[A-Z0-9]+:$1:; $final =~ s:CH([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^SH[A-Z0-9]+$/) {$initial =~ s:(SH)[A-Z0-9]+:$1:; $final =~ s:SH([A-Z0-9]+):$1:;} 
    elsif ($A[$i] =~ /^ZH[A-Z0-9]+$/) {$initial =~ s:(ZH)[A-Z0-9]+:$1:; $final =~ s:ZH([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^B[A-Z0-9]+$/) {$initial =~ s:(B)[A-Z0-9]+:$1:; $final =~ s:B([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^C[A-Z0-9]+$/) {$initial =~ s:(C)[A-Z0-9]+:$1:; $final =~ s:C([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^D[A-Z0-9]+$/) {$initial =~ s:(D)[A-Z0-9]+:$1:; $final =~ s:D([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^F[A-Z0-9]+$/) {$initial =~ s:(F)[A-Z0-9]+:$1:; $final =~ s:F([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^G[A-Z0-9]+$/) {$initial =~ s:(G)[A-Z0-9]+:$1:; $final =~ s:G([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^H[A-Z0-9]+$/) {$initial =~ s:(H)[A-Z0-9]+:$1:; $final =~ s:H([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^J[A-Z0-9]+$/) {$initial =~ s:(J)[A-Z0-9]+:$1:; $final =~ s:J([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^K[A-Z0-9]+$/) {$initial =~ s:(K)[A-Z0-9]+:$1:; $final =~ s:K([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^L[A-Z0-9]+$/) {$initial =~ s:(L)[A-Z0-9]+:$1:; $final =~ s:L([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^M[A-Z0-9]+$/) {$initial =~ s:(M)[A-Z0-9]+:$1:; $final =~ s:M([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^N[A-Z0-9]+$/) {$initial =~ s:(N)[A-Z0-9]+:$1:; $final =~ s:N([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^P[A-Z0-9]+$/) {$initial =~ s:(P)[A-Z0-9]+:$1:; $final =~ s:P([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^Q[A-Z0-9]+$/) {$initial =~ s:(Q)[A-Z0-9]+:$1:; $final =~ s:Q([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^R[A-Z0-9]+$/) {$initial =~ s:(R)[A-Z0-9]+:$1:; $final =~ s:R([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^S[A-Z0-9]+$/) {$initial =~ s:(S)[A-Z0-9]+:$1:; $final =~ s:S([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^T[A-Z0-9]+$/) {$initial =~ s:(T)[A-Z0-9]+:$1:; $final =~ s:T([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^W[A-Z0-9]+$/) {$initial =~ s:(W)[A-Z0-9]+:$1:; $final =~ s:W([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^X[A-Z0-9]+$/) {$initial =~ s:(X)[A-Z0-9]+:$1:; $final =~ s:X([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^Y[A-Z0-9]+$/) {$initial =~ s:(Y)[A-Z0-9]+:$1:; $final =~ s:Y([A-Z0-9]+):$1:;}
    elsif ($A[$i] =~ /^Z[A-Z0-9]+$/) {$initial =~ s:(Z)[A-Z0-9]+:$1:; $final =~ s:Z([A-Z0-9]+):$1:;}
    if ($initial ne $A[$i]) {
      $tone = $final;
      $final =~ s:([A-Z]+)[0-9]:$1:;
      $tone =~ s:[A-Z]+([0-9]):$1:;
      if (!(exists $py2ph{$initial}) or !(exists $py2ph{$final})) { print "1: no entry find for ", $A[$i], " ", $initial, " ", $final;  exit;}
      push(@entry, @{$py2ph{$initial}}); 
      @tmp = @{$py2ph{$final}};
      for($j = 0; $j < @tmp ; $j++) {$tmp[$j] = $tmp[$j].$tone;}
      push(@entry, @tmp); 
    }
    else {
      $tone = $A[$i];
      $A[$i] =~ s:([A-Z]+)[0-9]:$1:;   
      $tone =~ s:[A-Z]+([0-9]):$1:;
      if (!(exists $py2ph{$A[$i]})) { print "2: no entry find for ", $A[$i];  exit;}
      @tmp = @{$py2ph{$A[$i]}};
      for($j = 0; $j < @tmp ; $j++) {$tmp[$j] = $tmp[$j].$tone;}
      push(@entry, @tmp); 
    }
  } 
  print "@entry";
  print "\n";
}
