#!/usr/bin/perl


@ARGV ==1 || die "Usage: convert_states.pl dimension < states.txt > kaldi.am_gmm";

$dim = shift @ARGV;

$maxpdf = 0;
while(<>) {
    m/\<STATE\> (\d+) (\d+)/ || die "bad line $_";
    $pdf = $1; 
    if($pdf > $maxpdf) { $maxpdf = $pdf; }
    $nummix{$pdf} = $2;
    for($mix = 0; $mix < $nummix{$pdf}; $mix++) {
        $l = <>;
        chop $l;
        $weight{$pdf,$mix} = $l;
        $l = <>;
        @A = split(" ", $l);
        @A == $dim || die "Dimension mismatch.\n";
        for($d = 0; $d < $dim; $d++) {
            $mean{$pdf,$mix,$d} = $A[$d];
        }
        $l = <>;
        @A = split(" ", $l);
        @A == $dim || die "Dimension mismatch.\n";
        for($d = 0; $d < $dim; $d++) {
            $var{$pdf,$mix,$d} = $A[$d];
        }
    }
}

$numpdfs = $maxpdf+1;
print "<DIMENSION> 39 \n";
print "<NUMPDFS> $numpdfs\n";
for($pdf = 0; $pdf < $numpdfs; $pdf++) {
    defined  $nummix{$pdf} || die "No nummix defined for pdf = $pdf\n";
    print " <DiagGMMBegin>\n";
    $nm = $nummix{$pdf};
    print "  <WEIGHTS>  [ ";
    for($n = 0; $n < $nm; $n++) {  print "$weight{$pdf,$n} "; }
    print "]\n";
    print "  <MEANS_INVVARS>  [\n";
    for($n = 0; $n < $nm; $n++) { 
        for($d = 0; $d < $dim; $d++) {
            $val = $mean{$pdf,$n,$d} / $var{$pdf,$n,$d};
            print "$val ";
        }
        print "\n";
    }
    print "   ]\n";
    print "  <INV_VARS>  [\n";
    for($n = 0; $n < $nm; $n++) { 
        for($d = 0; $d < $dim; $d++) {
            $val = 1.0 / $var{$pdf,$n,$d};
            print "$val ";
        }
        print "\n";
    }
    print "   ]\n";
    print " <DiagGMMEnd>\n";
}

