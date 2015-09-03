#!/usr/bin/perl


if(@ARGV == 0) {
    die "Usage: integerize.pl symtab1 [symtab2].. < input > output\n";
   # integerize.pl takes as its arguments a symbol-table file like in OpenFst, and
   # converts the symbols on the standard input into integers.  Anything it does not 
   # recognize it leaves unchanged.
}

while( @ARGV > 0) {
    $symtab  = shift @ARGV;
    open(F, "<$symtab") || die "Error opening file $symtab\n";

    while(<F>){ 
        @A = split(" ", $_);
        @A == 2 || die "Bad line in symtab file $_: line $.\n";
        if(defined $sym2int{$A[0]}) {
            die "Multiply defined symbol $A[0]";
        }
        $sym2int{$A[0]} = $A[1];
    }
}

while(<STDIN>) {
    @A = split(" ", $_);
    foreach $a (@A) {
        $int = $sym2int{$a};
        if(defined $int) {
            print $int . " ";
        } else { 
            print $a . " ";
        }
    }
    print "\n";
}
