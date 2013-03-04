#!/bin/bash

# Creating a UEM decoding setup with CMU segmentation from Florian (Feb 15, 2013).
dummy_text=true
text=
#end of configuration

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ] ; then
  echo "Converts the CMU segmentation database file into a kaldi data directory for UEM decoding"
  echo ""
  echo "cmu_ume2kaldi_dir.sh <cmu-utt-database> <path-to-sph-files> <output-data-dir>"
  echo "example: cmu_ume2kaldi_dir.sh db-tag-eval-utt.dat /export/babel/data/106-tagalog/audio data/eval.uem"
fi

database=$1
audiopath=$2
datadir=$3

# 1. Create the segments file:
[ -f $database ] || echo "Database file $1 does not exist!" >&2 && exit 1;

cat $database | perl -pe 's:.+(BABEL):BABEL:; s:\}\s+\{FROM\s+: :; s:\}\s+\{TO\s+: :; s:\}.+::;' | \
  perl -ne 'split; 
            $utteranceID = @_[0]; 
            $utteranceID =~ s:[^_]+_[^_]+_[^_]+_::; 
            $utteranceID =~ s:([^_]+)_(.+)_(inLine|scripted):${1}_A_${2}:; 
            $utteranceID =~ s:([^_]+)_(.+)_outLine:${1}_B_${2}:; 
            $utteranceID .= sprintf ("_%06i", (100*@_[2])); 
            printf("%s %s %.2f %.2f\n", $utteranceID, @_[0], @_[1], @_[2]);' | sort > $datadir/segments


 # 2. Create the utt2spk file:

cut -f1 -d' ' $datadir/segments | \
  perl -ne 'chomp; m:([^_]+_[AB]).*:; print "$_ $1\n";' | \
  sort > $datadir/utt2spk

 # 3. Create the spk2utt file:

perl -ne '{chomp; split; $utt{@_[1]}.=" @_[0]";}
           END{foreach $spk (sort keys %utt) {
              printf("%s%s\n", $spk, $utt{$spk});
              }
           }' < $datadir/utt2spk | sort > $datadir/spk2utt

# 4. Create the wav.scp file:
sph2pipe=`which sph2pipe` || echo "Could not find sph2pipe binary. Add it to PATH" >&2 && exit 1;
for file in `cut -f 1 -d ' ' $datadir/segments` ; do
  [ -f  $audiopath/$file.sph ] || echo "Audio file $audiopath/$file.sph does not exist!" >&2 && exit 1;
  echo "$file $sph2pipe -f wav -p -c 1 $audiopath/$file.sph"
done | sort -u > $datadir/wav.scp


# 5. Create the text file:
if [ ! -z $text ] ; then
  cp $text $datadir/text || echo "Could not copy the source text file \"$text\" " && exit 1
elif $dummy_text ; then
  cut -f1 -d' ' segments | \
  sed -e 's/$/ IGNORE_TIME_SEGMENT_IN_SCORING/'  | \
  sort > $datadir/text
fi

# 6. reco2file_and_channel
(for f in $( cut -f 8 -d ' '  $database/wav.scp ) ; do p=`basename $f .sph`; echo $p $p 1; done) > reco2file_and_channel



