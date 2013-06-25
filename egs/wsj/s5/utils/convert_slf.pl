#!/usr/bin/perl

# Copyright 2013  Korbinian Riedhammer

# Convert a kaldi-lattice and convert it to HTK SLF format;  if given an output
# directory, each lattice will be put in an individual gzipped file.

use utf8;

binmode(STDIN, ":encoding(utf8)");
binmode(STDOUT, ":encoding(utf8)");

# defaults
$framerate=0.01;
$lmscale=1.0;
$acscale=1.0;
$wdpenalty=0.0;

if (@ARGV < 1 || @ARGV > 2) {
  print STDERR "Convert kaldi lattices to HTK SLF (v1.1) format.\n";
  print STDERR "Usage: convert_slf.pl [options] lat-file.txt [out-dir]\n";
  print STDERR "  e.g. lattice-word-align 'ark:gunzip -c lat.gz |' ark,t:- | $0 - slf/\n";
  print STDERR "Options regarding the SLF output:
  --lmscale x    LM weight (default: lmscale=$lmscale)
  --acscale x    Acoustic weight (default: acscale=$acscale)
  --wdpenalty x  Word insertion penalty (default: $wdpenalty)
  --framerate x  Frame rate to compute timing information (default: $framerate)
";

  exit 1;
}

while (@ARGV gt 0 and $ARGV[0] =~ m/^--/) {
  $param = shift @ARGV;
  if ($param eq "--lmscale") { $lmscale = shift @ARGV; }
  elsif ($param eq "--acscale") { $acscale = shift @ARGV; }
  elsif ($param eq "--wdpenalty") { $wdpenalty = shift @ARGV; }
  elsif ($param eq "--framerate") { $framerate = shift @ARGV; }
  else {
    print STDERR "Unknown option $param\n";
    exit 1;
  }
}

$outdir = "";
if (@ARGV == 2) {
  $outdir = pop @ARGV;
  unless (-d $outdir) {
    print STDERR "Could not find directory $outdir\n";
    exit 1;
  }
}


$utt = "";
@links = ();
%nodes = ();
%trace = ();

if ($outdir eq "") {
  open(FH, ">-") or die "Could not write to stdout (???)\n";
}

open (FI, $ARGV[0]) or die "Could not read from file\n";
binmode(FI, ":encoding(utf8)");

while(<FI>) {
  chomp;

  @A = split /\s+/;

  if (@A == 1 and $utt eq "") {
    # new lattice
    $utt = $A[0];
    $nodes{0} = 0.0;
    $trace{0} = 0;
  } elsif (@A == 1) {
    # do nothing with an accepting state
  } elsif (@A == 4) {
    # FSA arc
    ($s, $e, $w, $info) = @A;
    ($gs, $as, $ss) = split(/,/, $info);

    # kaldi saves -log, but HTK does it the other way round
    $gs *= -1;
    $as *= -1;
    
    # the state sequence is something like 1_2_4_56_45 so we remove all digits and count the _+1
    $ss =~ s/[0-9]*//g;
    $ss = 1 + length $ss;

    
    # we need the trace to compute the time segment
    $trace{$e} = $s;
    $nodes{$e} = $nodes{$s} + $ss * $framerate unless defined $nodes{$e}; # no not overwrite timing

    push @links, "S=$s\tE=$e\tW=$w\tv=0\ta=$as\tl=$gs";
  } elsif (@A == 0) {
    # print out the lattice;  open file handle first
    unless ($outdir eq "") {
      open(FH, "|-", "gzip -c > $outdir/$utt.lat.gz") or die "Could not write to $outdir/$utt.lat.gz\n";
      binmode(FH, ":encoding(utf8)");
    } 

    # header
    print FH "VERSION=1.1\n";
    print FH "UTTERANCE=$utt\n";
    print FH "lmscale=$lmscale\n";
    print FH "acscale=$acscale\n";
    print FH "N=".(keys %nodes)."\tL=".(@links)."\n";

    # nodes
    for $n (sort { $a <=> $b } keys %nodes) {
      printf FH "I=%d\tt=%.2f\n", $n, $nodes{$n};
    }

    # links/arks
    for $i (0 .. $#links) {
      print FH "J=$i\t".$links[$i]."\n";
    }

    print FH "\n";

    # close handle if it was a file
    close(FH) unless ($outdir eq "");

    # clear data
    $utt = "";
    @links = ();
    %nodes = ();
    %trace = ();
  }
}

if ($utt != "") {
  print STDERR "Last lattice was not printed as it might be incomplete?  Missing empty line?\n";
}

