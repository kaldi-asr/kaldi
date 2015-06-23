#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
max_seg_length=10
min_seg_length=2
min_sil_length=0.5
time_precision=0.05
special_symbol="<***>"
separator=";"
wer_cutoff=-1
# End configuration section.

set -e

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "This script takes the ctm file that corresponds to the data directory"
  echo "created by steps/cleanup/split_long_utterance.sh, works out a new"
  echo "segmentation and creates a new data directory for the new segmentation."
  echo ""
  echo "Usage: $0 [options] <ctm-file> <old-data-dir> <new-data-dir>"
  echo " e.g.: $0 train_si284_split.ctm \\"
  echo "                          data/train_si284_split data/train_si284_reseg"
  echo "Options:"
  echo "    --wer-cutoff            # ignore segments with WER higher than the"
  echo "                            # specified value. -1 means no segment will"
  echo "                            # be ignored."
  echo "    --max-seg-length        # maximum length of new segments"
  echo "    --min-seg-length        # minimum length of new segments"
  echo "    --min-sil-length        # minimum length of silence as split point"
  echo "    --time-precision        # precision for determining \"same time\""
  echo "    --special-symbol        # special symbol to be aligned with"
  echo "                            # inserted or deleted words"
  echo "    --separator             # separator for aligned pairs"
  exit 1;
fi

ctm=$1
old_data_dir=$2
new_data_dir=$3

for f in $ctm $old_data_dir/text.orig $old_data_dir/utt2spk \
  $old_data_dir/wav.scp $old_data_dir/segments; do
  if [ ! -f $f ]; then
    echo "$0: expected $f to exist"
    exit 1;
  fi
done

mkdir -p $new_data_dir/tmp/
cp -f $old_data_dir/wav.scp $new_data_dir
[ -f old_data_dir/spk2gender ] &&  cp -f $old_data_dir/spk2gender $new_data_dir

# Removes the overlapping region (in utils/split_long_utterance.sh we create
# the segmentation with overlapping region).
#
# Note that for each audio file, we expect its segments have been sorted in time
# ascending order (if we ignore the overlap).
cat $ctm | perl -e '
  $precision = $ARGV[0];
  @ctm = ();
  %processed_ids = ();
  $previous_id = "";
  while (<STDIN>) {
    chomp;
    my @current = split;
    @current >= 5 || die "Error: bad line $_\n";
    $id = join("_", ($current[0], $current[1]));
    @previous = @{$ctm[-1]};

    # Start of a new audio file.
    if ($previous_id ne $id) {
      # Prints existing information.
      if (@ctm > 0) {
        foreach $line (@ctm) {
          print "$line->[0] $line->[1] $line->[2] $line->[3] $line->[4]\n";
        }
      }

      # Checks if the ctm file is sorted.
      if (defined($processed_ids{$id})) {
        die "Error: \"$current[0] $current[1]\" has already been processed\n";
      } else {
        $processed_ids{$id} = 1;
      }

      @ctm = ();
      push(@ctm, \@current);
      $previous_id = $id;
      next;
    }

    $new_start = sprintf("%.2f", $previous[2] + $previous[3]);

    if ($new_start > $current[2]) {
      # Case 2: scans for a splice point.
      $index = -1;
      while (defined($ctm[$index])
             && $ctm[$index]->[2] + $ctm[$index]->[3] > $current[2]) {
        if ($ctm[$index]->[4] eq $current[4]
            && abs($ctm[$index]->[2] - $current[2]) < $precision
            && abs($ctm[$index]->[3] - $current[3]) < $precision) {
          pop @ctm for 2..abs($index);
          last;
        } else {
          $index -= 1;
        }
      }
    } else {
      push(@ctm, \@current);
    }
  }

  if (@ctm > 0) {
    foreach $line (@ctm) {
      print "$line->[0] $line->[1] $line->[2] $line->[3] $line->[4]\n";
    }
  }' $time_precision > $new_data_dir/tmp/ctm

# Creates a text file from the ctm, which will be used in Levenshtein alignment.
# Note that we remove <eps> in the text file.
cat $new_data_dir/tmp/ctm | perl -e '
  $previous_wav = "";
  $previous_channel = "";
  $text = "";
  while (<STDIN>) {
    chomp;
    @col = split;
    @col >= 5 || die "Error: bad line $_\n";
    if ($previous_wav eq $col[0]) {
      $previous_channel eq $col[1] ||
        die "Error: more than one channels detected\n";
      if ($col[4] ne "<eps>") {
        $text .= " $col[4]";
      }
    } else {
      if ($text ne "") {
        print "$previous_wav $text\n";
      }
      $text = $col[4];
      $previous_wav = $col[0];
      $previous_channel = $col[1];
    }
  }
  if ($text ne "") {
    print "$previous_wav $text\n";
  }' > $new_data_dir/tmp/text

# Computes the Levenshtein alignment.
align-text --special-symbol=$special_symbol --separator=$separator \
  ark:$old_data_dir/text.orig ark:$new_data_dir/tmp/text \
  ark,t:$new_data_dir/tmp/aligned.txt

# Creates new segmentation.
steps/cleanup/create_segments_from_ctm.pl \
  --max-seg-length $max_seg_length --min-seg-length $min_seg_length \
  --min-sil-length $min_sil_length \
  --separator $separator --special-symbol $special_symbol \
  --wer-cutoff $wer_cutoff \
  $new_data_dir/tmp/ctm $new_data_dir/tmp/aligned.txt \
  $new_data_dir/segments $new_data_dir/text

# Now creates the new utt2spk and spk2utt file.
cat $old_data_dir/utt2spk | perl -e '
  ($old_seg_file, $new_seg_file, $utt2spk_file_out) = @ARGV;
  open(OS, "<$old_seg_file") || die "Error: fail to open $old_seg_file\n";
  open(NS, "<$new_seg_file") || die "Error: fail to open $new_seg_file\n";
  open(UO, ">$utt2spk_file_out") ||
    die "Error: fail to open $utt2spk_file_out\n";
  while (<STDIN>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    $utt2spk{$col[0]} = $col[1];
  }
  while (<OS>) {
    chomp;
    @col = split;
    @col == 4 || die "Error: bad line $_\n";
    if (defined($wav2spk{$col[1]})) {
      $wav2spk{$col[1]} == $utt2spk{$col[0]} ||
        die "Error: multiple speakers detected for wav file $col[1]\n";
    } else {
      $wav2spk{$col[1]} = $utt2spk{$col[0]};
    }
  }
  while (<NS>) {
    chomp;
    @col = split;
    @col == 4 || die "Error: bad line $_\n";
    defined($wav2spk{$col[1]}) ||
      die "Error: could not find speaker for wav file $col[1]\n";
    print UO "$col[0] $wav2spk{$col[1]}\n";
  } ' $old_data_dir/segments $new_data_dir/segments $new_data_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $new_data_dir/utt2spk > $new_data_dir/spk2utt

utils/fix_data_dir.sh $new_data_dir

exit 0;
