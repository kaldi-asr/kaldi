#!/usr/bin/perl


# Usage: get_hmm_states.pl < MMF > states_file
# Format is:
# each mixture has 3 lines: mixture-weight, mean, variance.
# <STATE> A_s2_25 5 
#  7.630738e-02
#  1.340054e-01 8.793820e-01 3.381338e+00 4.707607e+00 2.738673e+00 2.130200e+00 1.505306e+00 5.386179e-01 -2.053278e+00 -2.369776e+00 -1.863615e+00 1.138925e-01 -5.315916e+00 -4.409675e-01 3.801185e-01 5.222713e-01 6.952143e-01 5.838995e-01 2.584392e-01 1.045933e-01 2.842196e-01 1.797329e-01 -2.383467e-01 -3.872046e-01 -1.208599e-01 -1.123077e+00 1.104437e-01 -4.043639e-01 -5.273581e-01 -5.055419e-01 -3.857868e-01 -4.783591e-02 -1.888570e-02 -2.677171e-01 2.246067e-01 2.122188e-01 2.536023e-01 5.772380e-03 5.641578e-01
#  1.210491e+01 2.887772e+01 1.444726e+01 2.377322e+01 2.243100e+01 3.112299e+01 3.070573e+01 2.498269e+01 2.727588e+01 2.047188e+01 1.985626e+01 9.705117e+00 4.483324e+01 8.487927e-01 1.378040e+00 1.394030e+00 1.389163e+00 1.826177e+00 2.960613e+00 2.533339e+00 1.901944e+00 1.711646e+00 2.056901e+00 1.904792e+00 1.298796e+00 1.229760e+00 1.435853e-01 2.742554e-01 2.303578e-01 3.856331e-01 3.890942e-01 5.537956e-01 5.028566e-01 4.502904e-01 5.239967e-01 4.440382e-01 3.432592e-01 2.598946e-01 2.945138e-01
#  1.673454e-01
# ...
# ...

while(<>) {
    if(m/\<BEGINHMM\>/) { # Avoid parsing lines like ~s "m_s4_6" that appear in HMM definitions.
        while(1) {
            $l = <>;
            chop $l;
            if($l eq "<ENDHMM>") {  last; }
        }
    }
    if(m/^\~s \"(.+)\"/) { # state macro begins.
        $macroname = $1;
        $l = <>;
        $l =~ m/\<NUMMIXES\> (\d+)/ || die "bad line (1) $l;";
        $nummix = $1;
        print "<STATE> $macroname $nummix\n";
        for($n = 1; $n <= $nummix; $n++) {
            $l = <>;
            $l =~ m/\<MIXTURE\> (\d+) (\S+)/ || die "bad line, n=$n (2) $l";
            $n == $1 || die "Mixture number mismatch.";
            $mixweight = $2;
            print "$mixweight\n";
            $l = <>;
            $l =~ m/\<MEAN\>/ || die "bad line (3) $."; # discard line e.g. <MEAN> 39
            $l = <>;
            print $l; # Just print the next line which is the mean.
            $l = <>;
            $l =~ m/\<VARIANCE\>/ || die "bad line (4) $."; # discard line e.g. <VARIANCE> 39
            $l = <>;
            print $l; # Just print the next line which is the variance.
            $l = <>;
            $l =~ m/GCONST/ || die "Unexpected line (5) $l (line $.)";
        }
    }
}

