#!/bin/bash -e

# Creating a UEM decoding setup with CMU segmentation from Florian (Feb 15, 2013).
dummy_text=true
text=
filelist=
#end of configuration

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ] ; then
  echo "$0: Converts the CMU segmentation database file into a kaldi data directory for UEM decoding"
  echo ""
  echo "cmu_ume2kaldi_dir.sh <cmu-utt-database> <path-to-sph-files> <output-data-dir>"
  echo "example: cmu_ume2kaldi_dir.sh db-tag-eval-utt.dat /export/babel/data/106-tagalog/audio data/eval.uem"
  echo "Was called with: $*"
  exit 1;
fi

database=$1
audiopath=$2
datadir=$3

echo $0 $@
mkdir -p $datadir
# 1. Create the segments file:
[ ! -f $database ] && echo "Database file $1 does not exist!"  && exit 1;

echo "Converting `basename $database` to kaldi directory $datadir "
cat $database | perl -pe 's:.+(BABEL):BABEL:; s:\}\s+\{FROM\s+: :; s:\}\s+\{TO\s+: :; s:\}.+::;' | \
  perl -ne '@K = split; 
            $utteranceID = @K[0]; 
            $utteranceID =~ s:[^_]+_[^_]+_[^_]+_::; 
            $utteranceID =~ s:([^_]+)_(.+)_(inLine|scripted):${1}_A_${2}:; 
            $utteranceID =~ s:([^_]+)_(.+)_outLine:${1}_B_${2}:; 
            $utteranceID .= sprintf ("_%06i", (100*@K[2])); 
            printf("%s %s %.2f %.2f\n", $utteranceID, @K[0], @K[1], @K[2]);' | sort > $datadir/segments

if [ ! -z $filelist ] ; then
  mv $datadir/segments $datadir/segments.full
  grep -F -f $filelist $datadir/segments.full > $datadir/segments

  l=`grep -v -F -f $filelist $datadir/segments.full | cut -f 2 -d ' ' | sort -u | wc -l`
  echo "Because of using filelist, $l files omitted"
fi


 # 2. Create the utt2spk file:

echo "Creating the $datadir/utt2spk file"
cut -f1 -d' ' $datadir/segments | \
  perl -ne 'chomp; m:([^_]+_[AB]).*:; print "$_ $1\n";' | \
  sort > $datadir/utt2spk

 # 3. Create the spk2utt file:

echo "Creating the $datadir/spk2utt file"
perl -ne '{chomp; @K=split; $utt{@K[1]}.=" @K[0]";}
           END{foreach $spk (sort keys %utt) {
              printf("%s%s\n", $spk, $utt{$spk});
              }
           }' < $datadir/utt2spk | sort > $datadir/spk2utt

# 4. Create the wav.scp file:
sph2pipe=`which sph2pipe || which $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe`
if [ $? -ne 0 ] ; then
  echo "Could not find sph2pipe binary. Add it to PATH"  
  exit 1;
fi
sox=`which sox`
if [ $? -ne 0 ] ; then
  echo "Could not find sox binary. Add it to PATH"  
  exit 1;
fi

echo "Creating the $datadir/wav.scp file"
(
  set -o pipefail
  for file in `cut -f 2 -d ' ' $datadir/segments` ; do
    if [ -f $audiopath/audio/$file.sph ] ; then
      echo "$file $sph2pipe -f wav -p -c 1 $audiopath/audio/$file.sph |"
    elif [ -f $audiopath/audio/$file.wav ] ; then
      echo "$file $sox $audiopath/audio/$file.wav -r 8000 -c 1 -b 16 -t wav - downsample |"
    else
      echo "Audio file $audiopath/audio/$file.sph does not exist!" >&2 
      exit 1
    fi
  done | sort -u > $datadir/wav.scp 
  if [ $? -ne 0 ] ; then 
    echo "Error producing the wav.scp file" 
    exit 1
  fi
) || exit 1 

l1=`wc -l $datadir/wav.scp | cut -f 1 -d ' ' `
echo "wav.scp contains $l1 files"
if [ ! -z $filelist ] ; then 
  l2=`wc -l $filelist | cut -f 1 -d ' '`
  echo "filelist `basename $filelist` contains $l2 files"

  if [ "$l1" -ne "$l2" ] ; then
    echo "WARNING: Not all files from the specified fileset made their way into wav.scp"
  fi
fi

# 5. Create the text file:
echo "Creating the $datadir/text file"
if [ ! -z $text ] ; then
  cp $text $datadir/text || echo "Could not copy the source text file \"$text\" " && exit 1
elif $dummy_text ; then
  cut -f1 -d' ' $datadir/segments | \
  sed -e 's/$/ IGNORE_TIME_SEGMENT_IN_SCORING/'  | \
  sort > $datadir/text
fi

# 6. reco2file_and_channel
echo "Creating the $datadir/reco2file_and_channel file"
(for f in $( cut -f 1 -d ' '  $datadir/wav.scp ) ; do echo $f $f "1"; done) > $datadir/reco2file_and_channel
echo "Everything done"



