#!/usr/bin/perl -w
# subs_en2fr.pl -retrieve key from hash
# loads E and F sides of bitext into arrays
# loads bitext into hash
# prints f side of selected e input to STDIN
# example:
# tail -n 3 /mnt/corpora/subs/OpenSubtitles2016.en-fr.en > s
# echo "I should do this in python" >> s
# subs_en2fr.pl /mnt/corpora/subs/OpenSubtitles2016.en-fr.en /mnt/corpora/subs/OpenSubtitles2016.en-fr.fr s
# should write:
# loaded /mnt/corpora/subs/OpenSubtitles2016.en-fr.en into arrray
#loaded /mnt/corpora/subs/OpenSubtitles2016.en-fr.fr into array
#loaded hash
#Vas-y franco.
#Tu m'as tué le dos !
#- Tu vas pas le faire.
#no match

use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 3 or croak "USAGE subs_en2fr.pl SRC_SUBS_EN_FILE SRC_SUBS_FR_FILE SELECTED_EN_SEGMENTS_FILE

subs_en2fr.pl /mnt/corpora/subs/OpenSubtitles2016.en-fr.en /mnt/corpora/subs/OpenSubtitles2016.en-fr.fr selected_segments.en
";
}

my ($e, $f, $s) = @ARGV;

open my $E, '<', "$e" or croak "could not open file $e for reading $!";
my @e = <$E>;
close $E;

print STDERR "loaded $e into arrray\n";

open my $F, '<', "$f" or croak "could not open file $f for reading $!";
my @f = <$F>;
close $F;

print STDERR "loaded $f into array\n";

my %e2f_hash = ();

for my $i (0..$#e) {
    chomp $e[$i];
    chomp $f[$i];
    $e2f_hash{$e[$i]} = $f[$i];
}

print STDERR "loaded hash\n";

open my $S, '<', "$s" or croak "could not open file $s for reading $!";

while ( my $line = <$S> ) {
    chomp $line;
    if ( defined $e2f_hash{$line} ) {
	print "$e2f_hash{$line}\n";
    } else {
	croak "no match for $line";
    }
}

close $S;
