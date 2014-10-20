#!/usr/bin/env perl
#
# Johns Hopkins University (Author : Gaurav Kumar)
#
# This script should be run from one directory above the current one
#
# Rough partitions that are needed are :
#
# ASR Train : 120k utterances
# ASR tune : 20k utterances
# ASR eval : 20k utterances
# MT train : 105k utterances
# MT tune : Same as the ASR eval (20k utterances)
# MT eval : 20k utterances
#
# This script tries to find the closest possible matches so that conversations
# belong in one single partition and hence there is no speaker/conversation
# overlap between data partitions

use Storable 'dclone';

$textfile="data/local/data/train_all/text";
$tmp="data/local/tmp";

open(T, "<", "$textfile") || die "Can't open text file";

$ongoingConv = "";
%tmpSplits = ();
@splitNumbers = (17455, 20000, 100000, 20000, 100000);
$splitId = 0;
%splits = ();

while (<T>) {
 	@myStringComponents = split(/\s/);
	@uttid = split('-', $myStringComponents[0]);
	$currentConv = $uttid[0];
	if ($currentConv eq $ongoingConv) {
		# Same conversation, add to current hash
		#print "Same conversation";
		$tmpSplits{$ongoingConv} += 1;
	}
	else {
		# New conversation intiated, first check if there are enough entries
		# in the hash
		#print $ongoingConv . " " . get_entries_hash(\%tmpSplits) . "\n";
		if (get_entries_hash(\%tmpSplits) > $splitNumbers[$splitId]) {
			print "Finished processing split " . $splitId . ". It contains " . get_entries_hash(\%tmpSplits) . " entries. \n";
			#$splits{$splitId} = keys %tmpSplits;
			@newArr = keys %tmpSplits;
			$splits{$splitId} = dclone(\@newArr);
			%tmpSplits = ();
			$splitId += 1;
		}
		$ongoingConv = $currentConv;
		$tmpSplits{$ongoingConv} = 1;
	}
}
# Put final tmpsplits in the right partition
@newArr = keys %tmpSplits;
$splits{$splitId} = dclone(\@newArr);
foreach (keys  %splits) {
	#print $_ , " ", $splits{$_}, "\n";
}
print "Finished processing split " . $splitId . ". It contains " . get_entries_hash(\%tmpSplits) . " entries. \n"; 

# Write splits to file 
foreach my $key ( keys %splits ) {
	open(S, ">$tmp/split-$key") || die "Can't open splitfile to write";
	foreach my $file ( @{$splits{$key}} ) {
		print $file, "\n";
		print S "$file\n" || die "Error writing to file";
	}
	close(S);
}

sub get_entries_hash() {
	my $inputHashRef = shift; 
	$total = 0;
	foreach (keys %{$inputHashRef})
  	{
		$total += $inputHashRef->{$_};
  	}
	return $total;
}

