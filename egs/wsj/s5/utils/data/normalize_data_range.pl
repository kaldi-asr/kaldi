#!/usr/bin/env perl

# This script is intended to read and write scp files possibly containing indexes for
# sub-ranges of features, like
# foo-123  bar.ark:431423[78:89]
# meaning rows 78 through 89 of the matrix located at bar.ark:431423.
#
# Its purpose is to normalize lines which have ranges on top of ranges, like
#
# foo-123  bar.ark:431423[78:89][3:4]
#
# This program interprets the later [] expression as a sub-range of the matrix returned by the first []
# expression; in this case, we'd get
#
# foo-123  bar.ark:431423[81:82]
#
# Note that these ranges are based on zero-indexing, and have a 'first:last'
# interpretation, so the range [0:0] is a matrix with one row.  And also note
# that column ranges are permitted, after row ranges, and the row range may be
# empty, e.g.

# foo-123  bar.ark:431423[81:82,0:13]
# or
# foo-123  bar.ark:431423[81:82,0:13]
#

# This program reads from the standard input (or command-line file or files),
# and writes to the standard output.


# This function combines ranges, either row or column ranges.  start1 and end1
# are the first range, and start2 and end2 are interpreted as a sub-range of the
# first range.  It is acceptable for either start1 and end1, or start2 and end2, to
# be empty.
# This function returns the start and end of the range, as an array.
sub combine_ranges {
  ($row_or_column, $start1, $end1, $start2, $end2) = @_;

  if ($start1 eq "" && $end1 eq "") {
    return ($start2, $end2);
  } elsif ($start2 eq "" && $end2 eq "") {
    return ($start1, $end1);
  } else {
    # For now this script doesn't support the case of ranges like [20:], even
    # though they are supported at the C++ level.
    if ($start1 eq "" || $start2 eq "" || $end1 eq "" || $end2 == "") {
      chop $line;
      print("normalize_data_range.pl: could not make sense of line $line\n");
      exit(1)
    }
    if ($start1 + $end2 > $end1) {
      chop $line;
      print("normalize_data_range.pl: could not make sense of line $line " .
            "[second $row_or_column range too large vs first range, $start1 + $end2 > $end1]\n");
      exit(1);
    }
    return ($start2+$start1, $end2+$start1);
  }
}


while (<>) {
  $line = $_;
  # we only need to do something if we detect two of these ranges.
  # The following regexp matches strings of the form ...[foo][bar]
  # where foo and bar have no square brackets in them.
  if (m/\[([^][]*)\]\[([^][]*)\]\s*$/) {
    $before_range = $`;
    $first_range = $1;   # e.g. '0:500,20:21', or '0:500', or ',0:13'.
    $second_range = $2;  # has same general format as first_range.
    if ($_ =~ m/concat-feats /) {
      # sometimes in scp files, we use the command concat-feats to splice together
      # two feature matrices.  Handling this correctly is complicated and we don't
      # anticipate needing it, so we just refuse to process this type of data.
      print "normalize_data_range.pl: this script cannot [yet] normalize the data ranges " .
        "if concat-feats was in the input data\n";
      exit(1);
    }
    print STDERR "matched: $before_range $first_range $second_range\n";
    if ($first_range !~ m/^((\d*):(\d*)|)(,(\d*):(\d*)|)$/) {
      print STDERR "normalize_data_range.pl: could not make sense of input line $_";
      exit(1);
    }
    $row_start1 = $2;
    $row_end1 = $3;
    $col_start1 = $5;
    $col_end1 = $6;

    if ($second_range !~ m/^((\d*):(\d*)|)(,(\d*):(\d*)|)$/) {
      print STDERR "normalize_data_range.pl: could not make sense of input line $_";
      exit(1);
    }
    $row_start2 = $2;
    $row_end2 = $3;
    $col_start2 = $5;
    $col_end2 = $6;

    ($row_start, $row_end) = combine_ranges("row", $row_start1, $row_end1, $row_start2, $row_end2);
    ($col_start, $col_end) = combine_ranges("column", $col_start1, $col_end1, $col_start2, $col_end2);


    if ($row_start ne "") {
      $range = "$row_start:$row_end";
    } else {
      $range = "";
    }
    if ($col_start ne "") {
      $range .= ",$col_start:$col_end";
    }
    print $before_range . "[" . $range . "]\n";
  } else {
    print;
  }
}

__END__

# Testing
# echo foo |  utils/data/normalize_data_range.pl -> foo
# echo 'foo[bar:baz]' |  utils/data/normalize_data_range.pl -> foo[bar:baz]
# echo 'foo[bar:baz][bin:bang]' |  utils/data/normalize_data_range.pl -> normalize_data_range.pl: could not make sense of input line foo[bar:baz][bin:bang]
# echo 'foo[10:20][0:5]' |  utils/data/normalize_data_range.pl -> foo[10:15]
# echo 'foo[,10:20][,0:5]' |  utils/data/normalize_data_range.pl -> foo[,10:15]
# echo 'foo[,0:100][1:15]' |  utils/data/normalize_data_range.pl -> foo[1:15,0:100]
# echo 'foo[1:15][,0:100]' |  utils/data/normalize_data_range.pl -> foo[1:15,0:100]
# echo 'foo[10:20][0:11]' |  utils/data/normalize_data_range.pl -> normalize_data_range.pl: could not make sense of line foo[10:20][0:11] [second row range too large vs first range, 10 + 11 > 20]
# echo 'foo[,10:20][,0:11]' |  utils/data/normalize_data_range.pl -> normalize_data_range.pl: could not make sense of line foo[,10:20][,0:11] [second column range too large vs first range, 10 + 11 > 20]
