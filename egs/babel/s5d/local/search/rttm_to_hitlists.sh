#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
flen=0.01
segments=
utt_table=
# End configuration section
echo $0 "$@"
. ./utils/parse_options.sh || exit 1;

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

if [ $# -ne 5 ] ; then
  echo "Usage: <rttm-file> <kwlist-file> <ecf-file> <word-dir> <output-file>"
  exit 1
fi

rttm=$1
kwlist=$2
ecf=$3
workdir=$4
output=$5

for f in $rttm $kwlist $ecf ; do
  [ ! -f $f ] && echo "File \"$f\" does not exist." && exit 1
done

mkdir -p $workdir

{
  echo '<kwslist kwlist_filename="" language="" system_id="">'
  echo '</kwslist>'
} > $workdir/kwslist.xml

kwseval=`which KWSEval`
if [ -z "$kwseval" ] ; then
  echo >&2 "KWSEval from F4DE tools not found"
  exit 1
fi

(
  set -x
  $kwseval -c -r $rttm -e $ecf -t $kwlist -s $workdir/kwslist.xml -f $workdir/
)

grep -E ",,MISS" $workdir/alignment.csv | \
  perl -e '
      binmode STDIN, ":utf8";
      binmode STDOUT, ":utf8";
      binmode STDERR, ":utf8";

      use Data::Dumper;
      $flen='$flen';
      %SEGMENTS=();
      if ((defined $ARGV[0]) && ( $ARGV[0] ne "" )) {
        open(F, $ARGV[0]) or die "Cannot open \"$ARGV[0]\"";
        while(<F>) {
          @entries = split(" ", $_);
          $entries[2] = int($entries[2]/$flen+0.5);
          $entries[3] = int($entries[3]/$flen+0.5);
          push @{$SEGMENTS{$entries[1]}}, [@entries];
        }
        close(F);
      }

      while(<STDIN>) {
        chomp;
        @entries_tmp = split(",", $_);
        @entries = ($entries_tmp[3],
                    $entries_tmp[1],
                    int($entries_tmp[5]/$flen + 0.5),
                    int($entries_tmp[6]/$flen + 0.5),
                    1.0
                   );

        $fid = $entries[1];
        $start = $entries[2];
        $end = $entries[3];

        if ((defined $ARGV[0]) && ( $ARGV[0] ne "" )) {
          $found = 0;
          foreach $entry ( @{$SEGMENTS{$fid}} ) {
            if (($start >= $entry->[2]) && ($end <= $entry->[3])) {
              $relstart = $start - $entry->[2];
              $relend = $end - $entry->[2];
              print join(" ", $entries[0], $entry->[0], $relstart, $relend, 1.0) . "\n";
              if ($found eq 1) {
                print STDERR "WARNING: Segments file generates duplicate hits for the entry";
                print STDERR join(" ", @entries_tmp) . "\n";
              }
              $found = 1;
            }
          }
          if ($found eq 0) {
            print STDERR "WARNING: Segments file does not allow for finding entry ";
            print STDERR join(" ", @entries_tmp) . "\n";
          }
        } else {
          print join(" ", @entries) . "\n";
        }
      }
  ' "$segments" | sort | {
  if [ -z "$utt_table" ]; then
    cat -
  else
    utils/sym2int.pl -f 2 $utt_table
  fi
} > $output
