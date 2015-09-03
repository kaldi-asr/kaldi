#!/usr/bin/perl

# e.g. mmf2trans.pl < /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm84/MMF

# Usage: mmf2trans.pl [ mmf] > trans
# Transition format (output of this program):

# <TRANSITION> A 5
#  0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
#  0.000000e+00 6.615417e-01 3.384583e-01 0.000000e+00 0.000000e+00
#  0.000000e+00 0.000000e+00 7.631182e-01 2.368818e-01 0.000000e+00
#  0.000000e+00 0.000000e+00 0.000000e+00 6.437724e-01 3.562277e-01
#  0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
# <TRANSITION> B 5
#  0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
#  0.000000e+00 6.934822e-01 3.065178e-01 0.000000e+00 0.000000e+00


# This program works by first storing the ~t macros as a hash from the
# string to the (string) entry; it then, for each ~h macro, works out
# the central phone and looks for either a ~t macro or a <TRANSP> entry;
# it keeps updated a hash from the central-phone to the transition string.
# at the end it prints these out.


# ReadTransition() reads text like the following
# from the standard input.  Its only argument is the first line 
# (with transp on it).
#<TRANSP> 5
# 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
# 0.000000e+00 6.615417e-01 3.384583e-01 0.000000e+00 0.000000e+00
# 0.000000e+00 0.000000e+00 7.631182e-01 2.368818e-01 0.000000e+00
# 0.000000e+00 0.000000e+00 0.000000e+00 6.437724e-01 3.562277e-01
# 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
# It returns the same text, from "5" (in this case) onward.

sub ReadNLines {
    my $n = shift @_;
    $ans = "";
    for(my $x = 0; $x < $n; $x++)  {
        $ans = $ans . <>;
    }
    return $ans;
}

while(<>) {
    if(m/~h \"(.+)\"/) {
        $phone = $1; # in context currently.
        if($phone =~ m:.+\-(.+)\+.+: ) { $phone = $1; } # Remove context. 
        elsif ($phone =~ m:.+\-(.+): ) { $phone = $1; }
        elsif ($phone =~ m:(.+)\+.+: ) { $phone = $1; }
        while(<>) {
            if(m/\<ENDHMM\>/) { last; } # no longer parsing this HMM.
            elsif(m/~t \"(.+)\"/) {
                $macroname = $1;
                defined $trans{$macroname} || die "No such macro $macroname";
                if(!defined $phone2trans{$phone}) {
                    $phone2trans{$phone} = $trans{$macroname};
                } else {
                    $phone2trans{$phone} eq $trans{$macroname} || print STDERR "Conflicting definitions for transition matrix for phone $phone: this conversion program will give you the wrong answer.\n";
                }
            } elsif (m/\<TRANSP\> (\d+)/) {
                $nlines = $1;
                if(!defined $phone2trans{$phone}) {
                    $phone2trans{$phone} = ReadNLines($nlines);
                } else {
                  $v = ReadNLines($nlines);
                  $phone2trans{$phone} eq $v || print STDERR "Conflicting definitions for transition matrix for phone $phone: this conversion program will give you the wrong answer: $phone2trans{$phone} versus $v\n";
                }
            }
        }
    }

    if(m/\~t \"(.+)\"/) { # defining a transition macro...
        $macroname = $1;
        ($tok,$n) = split(" ", <>); # Split the line like <TRANSP> 5
        $tok == "<TRANSP>" || die "Bad line $. in MMF\n";
        $trans{$macroname} = ReadNLines($n);
    }
}


foreach $phone (keys %phone2trans) {
    $tmatrix = $phone2trans{$phone};
    $nlines = split("\n", $phone2trans{$phone}); # counts the lines...
    if( $phone ne "sp" ) { # Don't print out short pause because Kaldi models
       # won't be able to handle the "see-through" phone (doesn't count for
       # context).
        print "<TRANSITION> $phone $nlines\n$tmatrix";
    }
}
