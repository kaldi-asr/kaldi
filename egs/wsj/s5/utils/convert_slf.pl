#!/usr/bin/perl

# Copyright 2014  Brno University of Technology (author Karel Vesely)
# Copyright 2013  Korbinian Riedhammer

# Convert a kaldi-lattice to HTK SLF format;  if given an output
# directory, each lattice will be put in an individual gzipped file.

# Node is represented by hash:
# { W=>[word], t=>[time], n_out_arcs=>[number_of_outgoing_arcs] };
# (Time internally represented as integer number of frames.)
#
# Link representation by hash:
# { S=>[start_node], E=>[end_node], W=>[word], v=>[0], a=>[acoustic_score], l=>[graph_score] }
# 
# Words are internally added both to links and nodes, we can choose where to print them.
# More common is to put words to nodes, which is the default. In that case applies a sanity
# check that all original links pointing to same node must have had same symbol.


use utf8;
use List::Util qw(max);

binmode(STDIN, ":encoding(utf8)");
binmode(STDOUT, ":encoding(utf8)");

# defaults
$framerate=0.01;
$wordtolink=0;

if (@ARGV < 1 || @ARGV > 2) {
  print STDERR "Convert kaldi lattices to HTK SLF (v1.1) format.\n";
  print STDERR "Usage: convert_slf.pl [options] lat-file.txt [out-dir]\n";
  print STDERR "  e.g. lattice-word-align 'ark:gunzip -c lat.gz |' ark,t:- | $0 - slf/\n";
  print STDERR "Options regarding the SLF output:
  --frame-rate x  Frame rate to compute timing information (default: $framerate)
  --word-to-link   Print the word symbols at links (default: words at nodes)
";

  exit 1;
}

while (@ARGV gt 0 and $ARGV[0] =~ m/^--/) {
  $param = shift @ARGV;
  if ($param eq "--frame-rate") { $framerate = shift @ARGV; }
  elsif ($param eq "--word-to-link") { $wordtolink = 1; }
  else {
    print STDERR "Unknown option $param\n";
    exit 1;
  }
}

$outdir = "";
if (@ARGV == 2) {
  $outdir = pop @ARGV;
  unless (-d $outdir) { system("mkdir -p $outdir"); }
  unless (-d $outdir) {
    print STDERR "Could not create directory $outdir\n";
    exit 1;
  }
}



$utt = "";
$latest_time;
@links = ();
%nodes = ();
%accepting_states = ();
%n_out_arcs = ();

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
    $nodes{0} = { W=>"!NULL", t=>0.0, n_out_arcs=>0 };
    $latest_time = 0.0;

  } elsif (@A == 1) {
    # accepting node without FST weight, store data for link to terminal super-state
    $accepting_states{$A[0]} = { W=>"!NULL", v=>0, a=>0, l=>0 };

  } elsif (@A == 2) {
    # accepting state with FST weight on it, again store data for the link
    ($s, $info) = @A;
    ($gs, $as, $ss) = split(/,/, $info);

    # kaldi saves -log, but HTK does it the other way round
    $gs *= -1;
    $as *= -1;

    # the state sequence is something like 1_2_4_56_45, get number of tokens after splitting by '_':
    $ss = scalar split(/_/, $ss);
    
    # update the end time
    die "Node $s not yet visited, is lattice sorted topologically? $utt" unless exists $nodes{$s};
    $time_end = $nodes{$s}{t} + $ss;
    if ($latest_time < $time_end) { $latest_time = $time_end; }

    # add the link data
    $accepting_states{$A[0]} = { W=>"!NULL", v=>0, a=>$as, l=>$gs };

  } elsif (@A == 4) {
    # FSA arc
    ($s, $e, $w, $info) = @A;
    ($gs, $as, $ss) = split(/,/, $info);

    # rename epsilons to null
    $w = "!NULL" if $w eq "<eps>";

    # kaldi saves -log, but HTK does it the other way round
    $gs *= -1;
    $as *= -1;
    
    # the state sequence is something like 1_2_4_56_45, get number of tokens after splitting by '_':
    $ss = scalar split(/_/, $ss);
    
    # keep track of the number of outgoing arcs for each node 
    # (later, we will connect sinks to the terminal state)
    $nodes{$s}{n_out_arcs} += 1;

    # keep track of timing
    die "Node $s not yet visited, is lattice sorted topologically? $utt" unless exists $nodes{$s};
    $time_end = $nodes{$s}{t} + $ss;
    if ($latest_time < $time_end) { $latest_time = $time_end; }

    # sanity check on already existing node
    if (exists $nodes{$e}) {
      die "Node $e has different words on incoming links, $w vs. ".$nodes{$e}{W}.". ".
          "Words can't be stored in the nodes, use --word-to-link, $utt.\n" 
       if ($w ne $nodes{$e}{W}) and not $wordtolink;
      die "Node $e previously stored with different time ".$nodes{$e}{t}." now $time_end, $utt.\n"
       if $time_end ne $nodes{$e}{t};
    }
    # add node; do not overwrite
    $nodes{$e} = { W=>$w, t=>$time_end, n_out_arcs=>0 } unless defined $nodes{$e};

    # add the link data
    push @links, { S=>$s, E=>$e, W=>$w, v=>0, a=>$as, l=>$gs };

  } elsif (@A == 0) { # end of lattice reading, we'll add terminal super-state, and print it soon...
    # find sinks
    %sinks = ();
    for $n (keys %nodes) { 
      $sinks{$n} = 1 if ($nodes{$n}{n_out_arcs} == 0);
    }

    # sanity check: lattices need at least one sink!
    if (scalar keys %sinks == 0) {
      print STDERR "Error: $utt does not have at least one sink node-- cyclic lattice??\n";
    }

    # add terminal super-state, as we need to add link with optional fst-weight from accepting state.
    $last_node = max(keys(%nodes)) + 1;
    $nodes{$last_node} = { W=>"!NULL", t=>$latest_time };
    # accepting states may contain weights
    for $accept (sort { $a <=> $b } keys %accepting_states) {
      %a = %{$accepting_states{$accept}};
      push @links, { S=>$accept, E=>$last_node, W=>$a{W}, v=>$a{v}, a=>$a{a}, l=>$a{l} };
    }
    # sinks that are not accepting states have no weights
    for $sink (sort { $a <=> $b } keys %sinks) {
      unless(exists($accepting_states{$sink})) {
        print STDERR "WARNING: detected sink node which is not accepting state in lattice $utt, incomplete lattice?\n";
        $a = \$accepting_states{$accept};
        push @links, { S=>$accept, E=>$last_node, W=>"!NULL", v=>0, a=>0, l=>0 };
      }
    }

    # print out the lattice;  open file handle first
    unless ($outdir eq "") {
      open(FH, "|-", "gzip -c > $outdir/$utt.lat.gz") or die "Could not write to $outdir/$utt.lat.gz\n";
      binmode(FH, ":encoding(utf8)");
    } 

    # header
    print FH "VERSION=1.1\n";
    print FH "UTTERANCE=$utt\n";
    print FH "N=".(keys %nodes)."\tL=".(@links)."\n";

    # nodes
    for $n (sort { $a <=> $b } keys %nodes) {
      if ($wordtolink) {
        printf FH "I=%d\tt=%.2f\n", $n, $nodes{$n}{t}*$framerate;
      } else {
        printf FH "I=%d\tW=%s\tt=%.2f\n", $n, $nodes{$n}{W}, $nodes{$n}{t}*$framerate;
      }
    }

    # links/arks
    for $i (0 .. $#links) {
      %l = %{$links[$i]}; # get hash representing the link...
      if ($wordtolink) {
        printf FH "J=$i\tS=%d\tE=%d\tW=%s\tv=%f\ta=%f\tl=%f\n", $l{S}, $l{E}, $l{W}, $l{v}, $l{a}, $l{l};
      } else {
        printf FH "J=$i\tS=%d\tE=%d\tv=%f\ta=%f\tl=%f\n", $l{S}, $l{E}, $l{v}, $l{a}, $l{l};
      }
    }

    print FH "\n";

    # close handle if it was a file
    close(FH) unless ($outdir eq "");

    # clear data
    $utt = "";
    @links = ();
    %nodes = ();
    %accepting_states = ();
  } else {
    die "Unexpected column number of input line\n$_";
  }
}

if ($utt != "") {
  print STDERR "Last lattice was not printed as it might be incomplete?  Missing empty line?\n";
}

