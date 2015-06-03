#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

if (@ARGV != 1 || !($ARGV[0] =~ m/^-?\d+\.?\d*$/ &&  $ARGV[0] >= 0)) {
  print STDERR "Usage: extend_segments.pl time-in-seconds <segments >segments.extended \n" .
       "e.g. extend_segments.pl 0.25 <segments.1 >segments.2\n" .
       "This command modifies a segments file, with lines like\n" .
       " <utterance-id> <recording-id> <start-time> <end-time>\n" .
       "by extending the beginning and end of each segment by a certain\n" .
       "length of time.  This script makes sure the output segments do not\n" .
       "overlap as a result of this time-extension, and that there are no\n" .
       "negative times in the output.\n";
  exit 1;
}

$extend = $ARGV[0];

@all_lines = ();

while (<STDIN>) {
  chop;
  @A = split(" ", $_);
  if (@A != 4) {
    die "invalid line in segments file: $_";
  }
  $line = @all_lines;  # current number of lines.
  ($utt_id, $reco_id, $start_time, $end_time) = @A;

  push @all_lines, [ $utt_id, $reco_id, $start_time, $end_time ]; # anonymous array.
  if (! defined $lines_for_reco{$reco_id}) {
    $lines_for_reco{$reco_id} = [ ];  # push new anonymous array.
  }
  push @{$lines_for_reco{$reco_id}}, $line;
}

foreach $reco_id (keys %lines_for_reco) {
  $ref = $lines_for_reco{$reco_id};
  @line_numbers = sort { ${$all_lines[$a]}[2] <=> ${$all_lines[$b]}[2] } @$ref;


  {
    # handle start of earliest segment as a special case.
    $l0 = $line_numbers[0];
    $tstart = ${$all_lines[$l0]}[2] - $extend;
    if ($tstart < 0.0) { $tstart = 0.0; }
    ${$all_lines[$l0]}[2] = $tstart;
  }
  {
    # handle end of latest segment as a special case.
    $lN = $line_numbers[$#line_numbers];
    $tend = ${$all_lines[$lN]}[3] + $extend;
    ${$all_lines[$lN]}[3] = $tend;
  }
  for ($i = 0; $i < $#line_numbers; $i++) {
    $ln = $line_numbers[$i];
    $ln1 = $line_numbers[$i+1];
    $tend = ${$all_lines[$ln]}[3]; # end of earlier segment.
    $tstart = ${$all_lines[$ln1]}[2]; # start of later segment.
    if ($tend > $tstart) {
      $utt1 = ${$all_lines[$ln]}[0];
      $utt2 = ${$all_lines[$ln1]}[0];
      print STDERR "Warning: for utterances $utt1 and $utt2, segments " .
        "already overlap; leaving these times unchanged.\n";
    } else {
      $my_extend = $extend;
      $max_extend =  0.5 * ($tstart - $tend);
      if ($my_extend > $max_extend) { $my_extend = $max_extend; }
      $tend += $my_extend;
      $tstart -= $my_extend;
      ${$all_lines[$ln]}[3] = $tend;
      ${$all_lines[$ln1]}[2] = $tstart;
    }
  }
}

# leave the numbering of the lines unchanged.
for ($l = 0; $l < @all_lines; $l++) {
  $ref = $all_lines[$l];
  ($utt_id, $reco_id, $start_time, $end_time) = @$ref;
  printf("%s %s %.2f %.2f\n", $utt_id, $reco_id, $start_time, $end_time);
}

__END__

# testing below.

# ( echo a1 A 0 1; echo a2 A 3 4; echo b1 B 0 1; echo b2 B 2 3 ) | local/extend_segments.pl 1.0
a1 A 0.00 2.00
a2 A 2.00 5.00
b1 B 0.00 1.50
b2 B 1.50 4.00
# ( echo a1 A 0 2; echo a2 A 1 3 ) | local/extend_segments.pl 1.0
Warning: for utterances a1 and a2, segments already overlap; leaving these times unchanged.
a1 A 0.00 2.00
a2 A 1.00 4.00
# ( echo a1 A 0 2; echo a2 A 5 6; echo a3 A 3 4 ) | local/extend_segments.pl 1.0
a1 A 0.00 2.50
a2 A 4.50 7.00
a3 A 2.50 4.50
