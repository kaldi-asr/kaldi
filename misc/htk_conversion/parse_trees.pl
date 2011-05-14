#!/usr/bin/perl

# Usage: parse_trees.pl < htk_tree > pre_kaldi_tree
# e.g. parse_trees.pl < /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees
# [the output is not a standard Kaldi format but one that
# I will write a Kaldi tool to parse].
 
# This program converts a tree in HTK format (e.g. 
# /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees
# on BUT system) to trees in a format described below that's more convenient
# to convert to Kaldi files.

# (2)
# The trees for each of the positions of each of the phones 
# (the positions are numbered from zero (offset of -2 from HTK state), 
# i.e. 0, 1, 2.

# A few lines of the output are:
# H 2  SE 2 [ i e E W @ ] {  CE H_s4_3   SE 2 [ A x ] {  CE H_s4_2   CE H_s4_1  }  } 
# N 0 CE N_s2_1
# N 1  SE 0 [ d ] {  CE N_s3_2   CE N_s3_1  } 


while(<>) {
    if(m/^QS (\S+) \{ (\S+) \}/) {
        # a line like this:
        # QS 'L_Vowel' { "A-*","x-*","u-*","U-*","X-*","I-*","i-*","o-*","e-*","O-*","E-*","Y-*","c-*","a-*","W-*","@-*" }
        $qname = $1;
        @set = split(",", $2);
        # e.g. of $2:
        #"A-*","x-*","u-*","U-*","X-*","I-*","i-*","o-*","e-*","O-*","E-*","Y-*","c-*","a-*","W-*","@-*"
        @set > 0 || die "Bad line (1) $_";
        @phoneset = ( ); 
        if($set[0] =~ m:\".+\-\*\":) { # left-context.
            $pos = 0; # position of context we're asking about, in Kaldi format.
            foreach $a (@set) {
                $a =~ m:\"(.+)\-\*\": || die "Bad line (2) $_";
                push @phoneset, $1;
            }
        } else { # Right-context.
            $pos = 2; # position of context we're asking about, in Kaldi format
            foreach $a (@set) {
                $a =~ m:\"\*\+(.+)\": || die "Bad line (3) $_";
                push @phoneset, $1;
            }
        }
        $question{$qname} = $pos . " [ " . join(" ",@phoneset) . " ]";
        # print STDERR "$qname -> $question{$qname}\n";
    } elsif(m/^$/) {
        next;
    } elsif(m/(.+)\[(\d+)\]/) {
        $phone = $1; $position = $2 - 2;
        $nextline = <>;
        if($nextline =~ m/^\s*\"(.+)\"\s*$/) { # no splits.
            $leaf = $1;
#e.g.:
#N[2]
#   "N_s2_1"
            print "$phone $position CE $leaf\n";
#e.g.:
#H[4]
#{
#   0                'R_Front'      -1      "H_s4_3" 
#  -1               'R_Middle'   "H_s4_1"   "H_s4_2" 
#}
        } elsif($nextline =~ /\{/) { # just "{" on a line by itself...
            $m = "";
            @pos2line = ( );
            while(1) { 
                $m = <>;
                if($m =~ m/\s*\}\s*$/) { #  "}" on its own line..
                    last;
                }
                @A = split(" ", $m);
                @A == 4 || die "Bad line $m: line $.\n";
                $pos2line[-$A[0]] = $m;
            }
            @pos2str = ( ); # Recursive, parenthesis-based representation of each line.
            # HERE.
            for($x = @pos2line-1; $x >= 0; $x--) {
                @A = split(" ", $pos2line[$x]);
                @A==4 || die "bad line [code error]\n"; 
                ($n,  $qname, $no, $yes) = @A;
                if($no =~ m:\-(\d+):) { # e.g. -1
                    $no_str = $pos2str[$1];
                } elsif ($no =~ m:\"(.+)\":) {
                    $no_str = " CE $1 ";  # e.g. "H_s3_2"
                } else { die "Bad line $pos2line[$x] or code error: before line $."; }
                if($yes =~ m:\-(\d+):) { # e.g. -1
                    $yes_str = $pos2str[$1];
                } elsif ($yes =~ m:\"(.+)\":) {
                    $yes_str = " CE $1 ";  # e.g. "CE H_s3_2"
                } else { die "Bad line $pos2line[$x] or code error: before line $."; }
                defined $question{$qname} || die "No such question $qname\n";
                $pos2str[$x] = "SE $question{$qname} { $yes_str $no_str } "; # yes before no in format we print.
            }
            $treestr = $pos2str[0]; 
            print "$phone $position $treestr\n"
        } else {
            die "Could not parse line $_ (1): line $.\n";
        }
    } else {
        die "Could not parse line $_ (2): line $.\n";
    }
}
