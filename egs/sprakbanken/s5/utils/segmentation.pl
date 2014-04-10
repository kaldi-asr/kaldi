#!/usr/bin/perl -w
# Copyright 2013  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This program is for segmentation of data, e.g. long telephone conversations,
# into short chunks.  The input (stdin) should be a sequence of lines like
# sw0-20348-A  0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 ...  2 2 0 0 0
# where there is a number for each frame and the numbers mean 0 for silence, 1
# for noise, laughter and other nonspeech events, and 2 for speech.  This will
# typically be derived from some kind of fast recognition (see
# ../steps/resegment_data.sh), followed by ali-to-phones --per-frame=true and
# then mapping phones to these classes 0, 1 and 2.
#
# The algorithm is as follows:
#  (1) Find contiguous sequences of classes 1 or 2 (i.e. speech and/or noise), with e.g.
#      "1 1 1 2 2" counted as a single contiguous sequence.  Each such sequence is an
#      initial segment.
#  (2) While the proportion of silence in the segments is less than $silence_proportion,
#      add a single silence frame to the left and right of each segment, as long
#      as this does not take us past the ends of the file or into another segment.  
#      At this point, do not merge segments.
#  (3) Merging segments:
#      Get a list of all boundaries between segments that ended up touching each other
#      during phase 2.  Sort them according to the number of silence frames at the boundary,
#      with those with the least silence to be processed first.  Go through the boundaries
#      in order, merging each pair of segments, as long as doing so does not create
#      a segment larger than $max_segment_length.
#  (4) Splitting excessively long segments:
#      For all segments that are longer than $hard_max_segment_length, split them equally
#      into the smallest number of pieces such that the pieces will be no longer than
#      $hard_max_segment_length.  Print a warning.
#  (5) Removing any segments that contain no speech.  (remove segments that have only silence
#      and noise.
#
#  By default, the utterance-ids will be of the form <RECORDING-ID>-<START-TIME>-<END-TIME>,
#  where <START-TIME> and <END-TIME> are measured 0.01 seconds, using fixed-width
#  integers with enough digits to print out all the segments (the number of digits being
#  decided per line of the input).  For instance, if the input recording-id was
#  sw0-20348-A, an example line of the "segments-file" output would be:
#   sw0-20348-A-00124-00298 sw0-20348-A 1.24 2.98
#  (interpreted as <UTTERANCE-ID> <RECORDING-ID> <START-TIME> <END-TIME>)
#  and the number of digits has to be that large because the same recording has
#  a segment something like
#   sw0-20348-A-13491-13606 sw0-20348-A 134.91 136.06
#  The "_" and "-" in the output are separately configurable by means of the
#  --first-separator and --second-separator options.  However, generally speaking,
#  it is safer to use "-" than, say, "_", because "-" appears very early in the
#  ASCII table, and using it as the separator will tend to ensure than when
#  you sort the utterances and the recording-ids they will sort the same way.
#  This matters because recording-ids will often equal speaker-ids, and Kaldi scripts
#  require that the utterance-ids and speaker-ids sort in the "same order".


use Getopt::Long;

$silence_proportion = 0.2; # The amount of silence at the sides of segments is
                           # tuned to give this proportion of silence.

$frame_shift = 0.01; # Affects the interpretation of the options such as max_segment_length,
                     # and the seconds in the "segments" file.
$max_segment_length = 15.0; # Maximum segment length while we are merging segments...
                            # it will not allow merging segments to make segments longer than this.
$hard_max_segment_length = 30.0; # A hard maximum on the segment length; it will
                                 # break segments to get below this, even if there is
                                 # no silence, and print a warning.
$first_separator = "-";   # separator between recording-id and start-time, in utterance-id.
$second_separator = "-";  # separator between start-time and end-time, in utterance-id.
$remove_noise_only_segments = "true";  # boolean option; if true,
                                       # remove segments that have no speech.


GetOptions('silence-proportion:f' => \$silence_proportion,
           'frame-shift:f' => \$frame_shift,
           'max-segment-length:f' => \$max_segment_length,
           'hard-max-segment-length:f' => \$hard_max_segment_length,
           'first-separator:s' => \$first_separator,
           'second-separator:s' => \$second_separator,
           'remove-noise-only-segments:s' => \$remove_noise_only_segments);

if (@ARGV != 0) {
  print STDERR "$0:\n" .
               "Usage: segmentation.pl [options] < per-frame-archive > segments-file\n" .
               "This program is called from steps/resegment_data.sh.  Please see\n" .
               "the extensive comment in the source.  Options:\n" .
               "--silence-proportion <float> (default: $silence_proportion)\n" .
               "--frame-shift <float> (default: $frame_shift, in seconds)\n" .
               "--max-segment-length <float> (default: $max_segment_length, in seconds)\n" .
               "--hard-max-segment-length (default: $hard_max_segment_length, in seconds)\n" .
               "--first-separator <string> (default: $first_separator), affects utterance-ids\n" .
               "--second-separator <string> (default: $second_separator), affects utterance-ids\n" .
               "--remove-noise-only-segments <true|false> (default: true)\n";
  exit 1;
}

($silence_proportion > 0.01 && $silence_proportion < 0.99) ||
  die "Invalid silence-proportion value '$silence_proportion'";
($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";
($max_segment_length > 1.0 && $max_segment_length < 100.0) ||
  die "Very strange max-segment-length value '$max_segment_length'";
($hard_max_segment_length > 4.0 && $hard_max_segment_length < 500.0) ||
  die "Very strange hard-max-segment-length value '$hard_max_segment_length'";
($hard_max_segment_length >= $max_segment_length) ||
  die "hard-max-segment-length may not be less than max-segment-length";
($remove_noise_only_segments eq 'false' ||
 $remove_noise_only_segments eq 'true') || 
  die "Option --remove-noise-only-segments takes args true or false";


sub get_initial_segments {
  # This operates on the global arrays @A, @S and @N.  It sets the elements of
  # @S to 1 if start of segment, and @E to 1 if end of segment, end of segment
  # being defined as one past the last frame in the segment.

  for (my $n = 0; $n < $N; $n++) {
    if ($A[$n] == 0) {
      if ($n > 0 && $A[$n-1] != 0) {
        $E[$n] = 1;
      }
    } else {
      if ($n == 0 || $A[$n-1] == 0) {
        $S[$n] = 1;
      }
    }
  }
  if ($A[$N-1] != 0) { # Handle the special case
    $E[$N] = 1;        # where the last frame of the file is silence or noise.
  }
}


sub set_silence_proportion {
  $num_nonsil_frames = 0;
  # Get number of frames that are inside segments.  Initially, this will
  # all be non-silence.
  $in_segment = 0;

  my @active_frames = (); # active_frames are segment start/end frames.
  for (my $n = 0; $n <= $N; $n++) {
    if ($n < $N && $S[$n] == 1) {
      $in_segment == 0 || die; 
      $in_segment = 1; 
      push @active_frames, $n;
    }
    if ($E[$n] == 1) { 
      $in_segment == 1 || die; 
      $in_segment = 0; 
      push @active_frames, $n;
    }
    if ($n < $N) {
      ($in_segment == ($A[$n] != 0 ? 1 : 0)) || die; # Just a check.
      if ($in_segment) { $num_nonsil_frames++; }
    }
  }
  $in_segment == 0 || die; # should not be still in a segment after file-end.
  if ($num_nonsil_frames == 0) {
    print STDERR "$0: warning: no segments found for recording $recording_id\n";
    return;
  }
  #(target-segment-frames - num-nonsil-frames) / target-segment-frames =  sil-proportion
  # -> target-segment-frames = (num-nonsil-frames) / (1 - sil-proportion).
  my $target_segment_frames = int($num_nonsil_frames / (1.0 - $silence_proportion));
  my $num_segment_frames = $num_nonsil_frames;
  while ($num_segment_frames < $target_segment_frames) {
    $changed = 0;
    for (my $i = 0; $i < @active_frames; $i++) {
      my $n = $active_frames[$i];
      if ($E[$n] == 1 && $n < $N && $S[$n] != 1) {
        # shift the end of this segment one frame to the right.
        $E[$n] = 0;
        $E[$n+1] = 1;
        $active_frames[$i] = $n + 1;
        $num_segment_frames++;
        $changed = 1;
      }
      if ($n < $N && $S[$n] == 1 && $n > 0 && $E[$n] != 1) {
        # shift the start of this segment one frame to the left
        $S[$n] = 0;
        $S[$n-1] = 1;
        $active_frames[$i] = $n - 1;
        $num_segment_frames++;
        $changed = 1;
      }
      if ($num_segment_frames == $target_segment_frames) {
        last;
      }
    }
    if ($changed == 0) { last; } # avoid an infinite loop.
  }
  if ($num_segment_frames < $target_segment_frames) {
    my $proportion = 
      ($num_segment_frames - $num_nonsil_frames) / $num_segment_frames;
    print STDERR "$0: warning: for recording $recording_id, only got a proportion " .
      "$proportion of silence frames, versus target $silence_proportion\n";
  }
}

sub merge_segments() {
  my @boundaries = ();
  my @num_silence_phones = (); # for any index into @S where there
                               # is a boundary between contiguous segments
                               # (i.e. an index which is both a segment-start
                               # and segment-end index), the number of silence
                               # frames at that boundary (i.e. at the end of the
                               # previous segment and the beginning of the next
                               # one.
  for ($n = 0; $n < $N; $n++) {
    if ($S[$n] == 1 && $E[$n] == 1) {
      push @boundaries, $n;
      my $num_sil = 0;
      my $p;
      # note: here we can count the silence phones without regard to the
      # segment boundaries, since we'll hit nonsilence before we get to
      # the end/beginning of these segments.
      for ($p = $n; $p < $N; $p++) {
        if ($A[$p] == 0) { $num_sil++; }
        else { last; }
      }
      for ($p = $n - 1; $p >= 0; $p--) {
        if ($A[$p] == 0) { $num_sil++; }
        else { last; }
      }
      $num_silence_phones[$n] = $p;
    }
  }

  # Sort on increasing number of silence-phones, so we join the segments with
  # the smallest amount of silence at the boundary first.
  my @sorted_boundaries = 
    sort { $num_silence_phones[$a] <=> $num_silence_phones[$b] } @boundaries;

  foreach $n (@sorted_boundaries) {
    # Join the segments only if the length of the resulting segment would
    # be no more than $max_segment_length.
    ($S[$n] == 1 && $E[$n] == 1) || die;
    my $num_frames = 2; # total number of frames in the two segments we'll be merging..
                        # start the count from 2 because the loops below do not
                        # count the 1st frame of the segment to the right and
                        # the last frame of the segment to the left.
    my $p;
    for ($p = $n + 1; $p <= @A && $E[$p] == 0; $p++) {
      $num_frames++;
    }
    $E[$p] == 1 || die;
    for ($p = $n - 1; $p >= 0 && $S[$p] == 0; $p--) {
      $num_frames++;
    }
    $S[$p] == 1 || die;
    if ($num_frames * $frame_shift <= $max_segment_length) {
      # Join this pair of segments.
      $S[$n] = 0;
      $E[$n] = 0;
    }
  }
}

sub split_long_segments {
  for (my $n = 0; $n < @A; $n++) {
    if ($S[$n] == 1) { # segment starts here...
      my $p;
      for ($p = $n + 1; $p <= @A; $p++) {
        if ($E[$p] == 1) { last; }
      }
      ($E[$p] == 1) || die;
      my $segment_length = $p - $n;
      my $max_frames = int($hard_max_segment_length / $frame_shift);
      if ($segment_length > $max_frames) {
        # The segment is too long, we need to split it.  First work out
        # how many pieces to split it into.
        # We divide and round up to nearest larger int.
        my $num_pieces = int(($segment_length / $max_frames) + 0.99999);
        my $segment_length_in_seconds = $segment_length * $frame_shift;
        print STDERR "$0: warning: for recording $recording_id, splitting segment of " .
          "length $segment_length_in_seconds seconds into $num_pieces pieces " .
          "(--hard-max-segment-length $hard_max_segment_length)\n";
        my $frames_per_piece = int($segment_length / $num_pieces);
        my $i;
        for ($i = 1; $i < $num_pieces; $i++) {
          my $q = $n + $i * $frames_per_piece;
          # Insert a segment boundary at frame $q.
          $S[$q] = 1;
          $E[$q] = 1;
        }
      }
      if ($p - 1 > $n) {
        $n = $p - 1; # avoids some redundant work.
      }
    }
  }
}

sub remove_noise_only_segments {
  for (my $n = 0; $n < $N; $n++) {
    if ($S[$n] == 1) { # segment starts here...
      my $p;
      my $saw_speech = 0;
      for ($p = $n; $p <= $N; $p++) {
        if ($E[$p] == 1 && $p != $n) { last; }
        if ($A[$p] == 2) { $saw_speech = 1; }
      }
      $E[$p] == 1 || die;
      if (! $saw_speech) { # There was no speech in this segment, so remove it.
        $S[$n] = 0;
        $E[$p] = 0;
      }
      if ($p - 1 > $n) {
        $n = $p - 1; # Avoid some redundant work.
      }
    }
  }
}

sub print_segments {
  # We also do some sanity checking here.
  my @segments = (); # each element will be a string start-time:end-time, in frames.

  $N == @S || die; # check array size.
  ($N+1) == @E || die; # check array size.

  my $max_end_time = 0;

  for (my $n = 0; $n < $N; $n++) {
    if ($E[$n] == 1 && $S[$n] != 1) {
      die "Ending segment before starting it: n=$n.\n";
    }
    if ($S[$n]) {
      my $p;
      for ($p = $n + 1; $p < $N && $E[$p] != 1; $p++) {
        $S[$p] && die; # should not start a segment again, before ending it.
      }
      $E[$p] == 1 || die;
      push @segments, "$n:$p"; # push the start/end times.
      $max_end_time = $p;
      if ($p < $N && $S[$p] == 1) { $n = $p - 1; }
      else { $n = $p; }
      # note: we increment $n again before the next loop instance.
    }
  }

  if (@segments == 0) {
    print STDERR "$0: warning: no segments for recording $recording_id\n";
  }

  # we'll be printing the times out in hundredths of a second (regardless of the
  # value of $frame_shift), and first need to know how many digits we need (we'll be
  # printing with "%05d" or similar, for zero-padding.
  $max_end_time_hundredths_second = int(100.0 * $frame_shift * $max_end_time);
  $num_digits = 1;
  my $i = 1;
  while ($i < $max_end_time_hundredths_second) {
    $i *= 10;
    $num_digits++;
  }
  $format_str = "%0${num_digits}d"; # e.g. "%05d"

  foreach $s (@segments) {
    my ($start,$end) = split(":", $s);
    ($end > $start) || die;
    my $start_seconds = sprintf("%.2f", $frame_shift * $start);
    my $end_seconds = sprintf("%.2f", $frame_shift * $end);
    my $start_str = sprintf($format_str, $start_seconds * 100);
    my $end_str = sprintf($format_str, $end_seconds * 100);
    my $utterance_id = "${recording_id}${first_separator}${start_str}${second_separator}${end_str}";
    print "$utterance_id $recording_id $start_seconds $end_seconds\n"; # <-- Here is where the output happens.
  }
}



while (<STDIN>) {
  @A = split; # split line on whitespace.
  if (@A <= 1) {
    print STDERR "$0: warning: invalid input line $_";
    next;
  }
  $recording_id = shift @A;  # e.g. sw0-12430
  for ($n = 0; $n < @A; $n++) {
    $a = $A[$n];
    if ($a != 0 && $a != 1 && $a != 2) {
      die "Invalid value $a: expecting 0, 1 or 2.  Line is: $_";
    }
    $A[$n] = 0 + $a; # cast to integer, might be a bit faster.
  }
  # The array @S will contain 1 if a segment starts there and 0
  # otherwise.  The array @E will contain 1 if a segment ends there
  # and 0 otherwise.
  $N = @A; # number of elements in @A.  Used globally.
  @S = (0) x $N;         # 0 repeated $N times.
  @E = (0) x ($N + 1);   # 0 repeated $N+1 times (one more since if the last frame is
                         # in a segment, the end-marker goes one past that, at index $N.)

  get_initial_segments();       # stage (1) in the comment above.
  set_silence_proportion();     # stage (2)
  merge_segments();             # stage (3)
  split_long_segments();        # stage (4)
  if ($remove_noise_only_segments eq 'true') {
    remove_noise_only_segments(); # stage (5)
  }
  print_segments();
}

