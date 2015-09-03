#!/usr/bin/perl

# see README for example of usage.  This produces "empty" trees for the
# context-independent phones.  Input is MMF; output is trees in same format
# as produced by parse_trees.pl


while(<>) {
    if(m/\~h \"([^\-\+]+)\"/) {
        $phone = $1;  # This is a context-independent phone that has its own HMM.
        $l = <>;
        chop $l;
        $l eq "<BEGINHMM>" || die "Unexpected line $l\n";
        $l = <>;
        $l =~ m:<NUMSTATES> (\d+): || die "Unexpected line $l\n";
        $nstates = $1;
        for($n = 2; $n < $nstates; $n++) {
            $l = <>;
            $l =~ m/<STATE> (\d+)/ && $1 == $n || die "Unexpected line $l\n";
            $l = <>;
            $l =~ m/\~s \"(.+)\"/ || die "Unexpected line $l (perhaps your silence model does not have macros for states?)\n";
            $macroname = $1;
            $n2 = $n - 2; # The id we will use in the Kaldi code.
            if($phone ne "sp") { # We're omitting sp because can't be put in Kaldi format.
                print "$phone $n2 CE $macroname\n"; 
            }
        }
    }
}

