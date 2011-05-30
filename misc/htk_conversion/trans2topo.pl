#!/usr/bin/perl

# Converts intermediate transition-model format into Kaldi "topology" format (where
# the transitions become the "default" transition values in the topology; when we
# initialize the actual transition-model these will be used...

print "<Topology>\n";

while(<>) {
    if(m/\<TRANSITION\> (\S+) (\d+)/) {
        $phone = $1;
        $numstates = $2; # Normally 5.  Two are dummy states.
        print "<TopologyEntry> <ForPhones> $phone </ForPhones>\n";
        for($n = 1; $n <= $numstates; $n++) {
            $l = <>;
            @A = split(" ", $l);
            @A == $numstates || die "Bad line $l: line $.";
            if($n == 1) {
                if($A[1] != 1.0) {
                    print STDERR "Warning: phone $phone seems not to be normal topology: result may not be correct.\n";
                }
            } else {
                $nm2 = $n-2; # Kaldi-numbered state, 2 less than HTK one.
                if($n < $numstates) {
                  print " <State> $nm2 <PdfClass> $nm2\n";
                } else {
                  print " <State> $nm2\n";
                }
                # The next few lines are just a sanity check-- that we have the "normal" topology.
                for($p = 0; $p < $numstates; $p++) {
                    if($A[$p] != 0) {
                        $deststate = $p-1; # in kaldi numbering.
                        print "  <Transition> $deststate $A[$p]\n";
                    }
                }
                print " </State>\n";
            }
        }
        print "</TopologyEntry>\n";
    } else { 
        die "Bad line $_: line $.\n";
    }
}

print "</Topology>\n";


