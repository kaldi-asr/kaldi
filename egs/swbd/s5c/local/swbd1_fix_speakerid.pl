#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

# Author: Peng Qi (pengqi@cs.stanford.edu)
# This script maps Switchboard speaker IDs to the true physical speakers
# and fixes the utterances IDs accordingly. Expected to be run one level of
# directory above.

sub trim {
    (my $s = $_[0]) =~ s/^\s+|\s+$//g;
    return $s;        
}

if ($#ARGV != 1) {
	print "Usage: swbd1_fix_speakerid.pl <swbd-conv-tab-file> <data-dir>\n";
	print "E.g.:  swbd1_fix_speakerid.pl /datasets/SWBD1Transcripts/tables/conv.tab data/train\n";
}

$tab_file = $ARGV[0];
$dir = $ARGV[1];

%conv_to_spk = ();

open(my $conv_tab, '<', $tab_file) or die "Could not open '$tab_file' $!\n";
 
while (my $line = <$conv_tab>) {
  chomp $line;
 
  my @fields = split "," , $line;
  #$fields[0] = trim($fields[0]);
  $fields[2] = trim($fields[2]);
  $fields[3] = trim($fields[3]);
  $conv_to_spk{'sw0' . $fields[0] . '-A'} = $fields[2];
  $conv_to_spk{'sw0' . $fields[0] . '-B'} = $fields[3];
}

close($conv_tab);

# fix utt2spk

%missingconv = ();

open(my $utt2spk, '<', $dir . '/utt2spk') or die "Could not open '$dir/utt2spk' $!\n";
open(my $utt2spk_new, '>', $dir . '/utt2spk.new');

while (my $line = <$utt2spk>) {
  chomp $line;

  my @fields = split " " , $line;
  my $convid = substr $fields[0], 0, 9;
  
  if (exists $conv_to_spk{ $convid }) {
    my $spkid = $conv_to_spk{ $convid };
    $spkid = "sw" . $spkid;
    my $newuttid = $spkid . '-' . (substr $fields[0], 2);

    print $utt2spk_new "$newuttid $spkid\n";
  } else {
    my $convid = substr $convid, 3, 4;
    $missingconv{$convid} = 1;
    
    print $utt2spk_new $fields[0]." ".$fields[1]."\n";
  }
}

close($utt2spk);
close($utt2spk_new);

foreach my $conv (keys %missingconv) {
  print "Warning: Conversation ID '$conv' not found in conv.tab, retaining old speaker IDs\n"
}

# fix segments and text

foreach my $file ('segments','text') {
  open(my $oldfile, '<', "$dir/$file") or die "Could not open '$dir/$file' $!\n";
  open(my $newfile, '>', "$dir/$file.new");

  while (my $line = <$oldfile>) {
    chomp $line;

    my $convid = substr $line, 0, 9;
    if (exists $conv_to_spk{$convid}) {
      my $spkid = $conv_to_spk{$convid};
      print $newfile "sw$spkid-" . (substr $line, 2) . "\n";
    } else {
      print $newfile "$line\n";
    }
  }
}
