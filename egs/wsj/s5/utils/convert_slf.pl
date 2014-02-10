#!/usr/bin/perl

# Copyright 2014  Brno University of Technology (author Karel Vesely)
# Copyright 2013  Korbinian Riedhammer

# Convert a kaldi-lattice to HTK SLF format;  if given an output
# directory, each lattice will be put in an individual gzipped file.

# Internal representation of nodes, links:
# node hash:
# { W=>[word], t=>[time], n_out_arcs=>[number_of_outgoing_arcs] };
# (Time internally represented as integer number of frames.)
# link hash:
# { S=>[start_node], E=>[end_node], W=>[word], v=>[0], a=>[acoustic_score], l=>[graph_score] }
# 
# The HTK output supports:
# - words on links [default],
#   - simpler, same as in kaldi lattices, node-ids in output correspond to kaldi lattices
# - words on nodes,
#   - apart from original nodes, there are extra nodes containing the words.
#   - each original ark is replaced by word-node and two links, connecting it with original nodes.


use utf8;
use List::Util qw(max);

binmode(STDIN, ":encoding(utf8)");
binmode(STDOUT, ":encoding(utf8)");

# defaults
$framerate=0.01;
$wordtonode=0;

$usage="Convert kaldi lattices to HTK SLF (v1.1) format.\n".
       "Usage: convert_slf.pl [options] lat-file.txt [out-dir]\n".
       "  e.g. lattice-align-words lang/phones/word_boundary.int final.mdl 'ark:gunzip -c lat.gz |' ark,t:- | utils/int2sym.pl -f 3 lang/words.txt | $0 - slf/\n".
       "\n".
       "Options regarding the SLF output:\n".
       "  --frame-rate x  Frame rate to compute timing information (default: $framerate)\n".
       "  --word-to-node  Print the word symbols on nodes (adds extra nodes+links; default: words at links)\n".
       "\n";

# parse options
while (@ARGV gt 0 and $ARGV[0] =~ m/^--/) {
  $param = shift @ARGV;
  if ($param eq "--frame-rate") { $framerate = shift @ARGV; }
  elsif ($param eq "--word-to-node") { $wordtonode = 1;}
  else {
    print STDERR "Unknown option $param\n";
    print STDERR;
    print STDERR $usage;
    exit 1;
  }
}

# check positional arg count
if (@ARGV < 1 || @ARGV > 2) {
  print STDERR $usage;
  exit 1;
}

# store gzipped lattices individually to outdir:
$outdir = "";
if (@ARGV == 2) {
  $outdir = pop @ARGV;
  unless (-d $outdir) { system("mkdir -p $outdir"); }
  unless (-d $outdir) {
    print STDERR "Could not create directory $outdir\n";
    exit 1;
  }
}
# or we'll print lattices to stdout:
if ($outdir eq "") {
  open(FH, ">-") or die "Could not write to stdout (???)\n";
}


### parse kaldi lattices:

$utt = "";
$arc = 0;
$latest_time = 0.0;
@links = ();
%nodes = ();
%nodes_extra = ();
%accepting_states = ();

open (FI, $ARGV[0]) or die "Could not read from file\n";
binmode(FI, ":encoding(utf8)");

while(<FI>) {
  chomp;

  @A = split /\s+/;

  if (@A == 1 and $utt eq "") {
    # new lattice
    $utt = $A[0];
    $nodes{0} = { W=>"!NULL", t=>0.0, n_out_arcs=>0 }; #initial node

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

  } elsif (@A == 4 or @A == 3) {
    # FSA arc
    ($s, $e, $w, $info) = @A;
    if ($info ne "") {
      ($gs, $as, $ss) = split(/,/, $info);
    } else {
      $gs = 0; $as = 0; $ss = "";
    }

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
      die "Node $e previously stored with different time ".$nodes{$e}{t}." now $time_end, $utt.\n"
       if $time_end ne $nodes{$e}{t};
    }

    # store internal representation of the arc
    if (not $wordtonode) {
      # The words on links, the lattice keeps it's original structure,
      # add node; do not overwrite
      $nodes{$e} = { t=>$time_end, n_out_arcs=>0 } unless defined $nodes{$e};
      # add the link data
      push @links, { S=>$s, E=>$e, W=>$w, v=>0, a=>$as, l=>$gs };

    } else {
      # The problem here was that, if we have a node with several incoming links,
      # the links can have different words on it, so we cannot simply put word from 
      # link into the node.
      #
      # The simple solution is:
      # each FST arc gets replaced by extra node with word and two links,
      # connecting it with original nodes.
      #
      # The lattice gets larger, and it is good to minimize the lattice during importing.
      #
      # During reading the FST, we don't know how many nodes there are in total, 
      # so the extra nodes are stored separately, indexed by arc number, 
      # and links have flags describing which type of node are they connected to.

      # add 'extra node' containing the word:
      $nodes_extra{$arc} = { W=>$w, t=>$time_end };
      # add 'original node'; do not overwrite
      $nodes{$e} = { W=>"!NULL", t=>$time_end, n_out_arcs=>0 } unless defined $nodes{$e};
      
      # add the link from 'original node' to 'extra node'
      push @links, { S=>$s, E=>$arc, W=>$w, v=>0, a=>$as, l=>$gs, to_extra_node=>1 };
      # add the link from 'extra node' to 'original node'
      push @links, { S=>$arc, E=>$e, W=>$w, v=>0, a=>0, l=>0, from_extra_node=>1 };
   
      # increase arc counter 
      $arc++;
    }

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

    # add terminal super-state,
    $last_node = max(keys(%nodes)) + 1;
    $nodes{$last_node} = { W=>"!NULL", t=>$latest_time };

    # connect all accepting states with terminal super-state,
    for $accept (sort { $a <=> $b } keys %accepting_states) {
      %a = %{$accepting_states{$accept}};
      push @links, { S=>$accept, E=>$last_node, W=>$a{W}, v=>$a{v}, a=>$a{a}, l=>$a{l} };
    }

    # connect also all sinks that are not accepting states,
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

    if (not $wordtonode) {
      # print lattice with words on links:
      
      # header
      print FH "VERSION=1.1\n";
      print FH "UTTERANCE=$utt\n";
      print FH "N=".(keys %nodes)."\tL=".(@links)."\n";

      # nodes
      for $n (sort { $a <=> $b } keys %nodes) {
        printf FH "I=%d\tt=%.2f\n", $n, $nodes{$n}{t}*$framerate;
      }

      # links/arks
      for $i (0 .. $#links) {
        %l = %{$links[$i]}; # get hash representing the link...
        printf FH "J=$i\tS=%d\tE=%d\tW=%s\tv=%f\ta=%f\tl=%f\n", $l{S}, $l{E}, $l{W}, $l{v}, $l{a}, $l{l};
      }

    } else {
      # print lattice with words in the nodes:

      # header
      print FH "VERSION=1.1\n";
      print FH "UTTERANCE=$utt\n";
      print FH "N=".(scalar(keys(%nodes))+scalar(keys(%nodes_extra)))."\tL=".(@links)."\n";

      # number of original nodes, offset of extra_nodes
      $node_id_offset = scalar keys %nodes;

      # nodes
      for $n (sort { $a <=> $b } keys %nodes) {
        printf FH "I=%d\tW=%s\tt=%.2f\n", $n, $nodes{$n}{W}, $nodes{$n}{t}*$framerate;
      }
      # extra nodes
      for $n (sort { $a <=> $b } keys %nodes_extra) {
        printf FH "I=%d\tW=%s\tt=%.2f\n", $n+$node_id_offset, $nodes_extra{$n}{W}, $nodes_extra{$n}{t}*$framerate;
      }

      # links/arks
      for $i (0 .. $#links) {
        %l = %{$links[$i]}; # get hash representing the link...
        if ($l{from_extra_node}) { $l{S} += $node_id_offset; }
        if ($l{to_extra_node}) { $l{E} += $node_id_offset; }
        printf FH "J=$i\tS=%d\tE=%d\tv=%f\ta=%f\tl=%f\n", $l{S}, $l{E}, $l{v}, $l{a}, $l{l};
      }
    }

    print FH "\n";

    # close handle if it was a file
    close(FH) unless ($outdir eq "");

    # clear data
    $utt = "";
    $arc = 0;
    $latest_time = 0.0;
    @links = ();
    %nodes = ();
    %nodes_extra = ();
    %accepting_states = ();
  } else {
    die "Unexpected column number of input line\n$_";
  }
}

if ($utt != "") {
  print STDERR "Last lattice was not printed as it might be incomplete?  Missing empty line?\n";
}

