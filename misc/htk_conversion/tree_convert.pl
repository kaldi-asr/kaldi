#!/usr/bin/perl

# Converts intermediate representation of tree into Kaldi-format ContextDependency
# object.    Assumes triphone.

if (@ARGV != 2) {
    die "Usage: tree_convert.pl phone2len.txt tree.txt > kaldi.tree\n";
}

($phone2len, $tree_in) = @ARGV;
open(P, "<$phone2len") || die "Opening file $phone2len";
$maxphone = 0;
while(<P>) {
    @A = split(" ", $_);
    @A == 2 || die "bad phone2len file: line is $_\n";
    $len{$A[0]} = $A[1];
    if($A[0] > $maxphone) { $maxphone = $A[0]; }
}
open(T, "<$tree_in") || die "Opening tree file $tree_in";
while(<T>) {
    @A = split(" ", $_);
    $phone = shift @A;
    $pos = shift @A;
    $tree{$phone,$pos} = join(" ", @A);
}

# standard triphone settings:
$N = 3;
$P = 1;
print "ContextDependency $N $P\n";
$np = $maxphone+1; 
# printing out to-pdf map.. 1==split-on-central-position;
# $np is size of array in table-event-map.
print "ToPdf TE 1 $np (\n"; 
for($p = 0; $p < $np; $p++) {
    if(!defined $len{$p}) { # probably eps.
        print "NULL\n";
    } else {
        print " TE -1 $len{$p} (\n";  # table-event-map splitting on pdf-class == hmm-position.
        for($pos = 0; $pos < $len{$p}; $pos++) { # for each HMM-position (0,1,2)
            $treestr = $tree{$p,$pos};
            defined $treestr || die "No tree defined for phone=$p, pos=$pos\n";
            print "  $treestr\n";
            # E.g.: treestr = ( <Q> -1 ( 40 42 10 30 6 34 29 31 ) ( <Q> -1 ( 10 30 6 31 ) ( <Q> 1 ( 36 0 ) ( <L> 507 ) ( <L> 506 ) ) ( <L> 505 ) ) ( <Q> -1 ( 40 10 30 6 34 29 31 18 43 9 12 39 25 4 20 ) ( <L> 504 ) ( <Q> -1 ( 22 ) ( <L> 503 ) ( <Q> -1 ( 26 7 ) ( <L> 502 ) ( <Q> 1 ( 37 ) ( <L> 501 ) ( <L> 500 ) ) ) ) ) ) 
            # First map the position to a "kaldi-format" position whose number starts form zero,
            # by adding P.
        }
        print " )\n";
    }
}


print ")\n";
print "EndContextDependency\n";

