#!/usr/bin/perl

while(<>) {
    if(m/\<TRANSITION\> (\S+) (\d+)/) {
        print;
        $phone = $1;
        $nl = $2;
        for($x = 0; $x < $nl; $x++) {
            $l = <>;
            if($x > 0 && $x < $nl-1) {
                @A = split(" ", $l);
                @A == $nl || die "bad line $l";
                $removed = 0;
                for($y = 0; $y < $nl; $y++) {
                    if($A[$y] != 0.0 && 
                       $y != $x && $y != $x+1) { # not self-loop or trans to
                        # next state.
                        $removed += $A[$y];
                        $A[$y] = 0.0;
                    }
                }
                if($removed != 0.0) {
                    print STDERR "Removed prob mass $removed from phone $phone\n";
                }
                for($y = 0; $y < $nl; $y++) {
                    $A[$y] *= 1.0 / (1.0 - $removed);
                }
                print join(" ",@A) . "\n";
            } else {
                print $l; # no info on 1st and last lines, just print them.
            }
        }
    } else {
        die  "bad line $_: line $.";
    }
}
