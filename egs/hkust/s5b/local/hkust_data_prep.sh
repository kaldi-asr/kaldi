#!/bin/bash


. path.sh

if [ $# != 2 ]; then
   echo "Usage: hkust_data_prep.sh AUDIO_PATH TEXT_PATH"
   exit 1;
fi

HKUST_AUDIO_DIR=$1
HKUST_TEXT_DIR=$2

train_dir=data/local/train
dev_dir=data/local/dev


case 0 in    #goto here
    1)
;;           #here:
esac

mkdir -p $train_dir
mkdir -p $dev_dir

#data directory check
if [ ! -d $HKUST_AUDIO_DIR ] || [ ! -d $HKUST_TEXT_DIR ]; then
  echo "Error: run.sh requires two directory arguments"
  exit 1;
fi

#find sph audio file for train dev resp.
find $HKUST_AUDIO_DIR -iname "*.sph" | grep -i "audio/train" > $train_dir/sph.flist
find $HKUST_AUDIO_DIR -iname "*.sph" | grep -i "audio/dev" > $dev_dir/sph.flist

n=`cat $train_dir/sph.flist $dev_dir/sph.flist | wc -l`
[ $n -ne 897 ] && \
  echo Warning: expected 897 data data files, found $n


#Transcriptions preparation

#collect all trans, convert encodings to utf-8,
find $HKUST_TEXT_DIR -iname "*.txt" | grep -i "trans/train" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; } 
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:; 
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5; 
        for($n = 3; $n < @A; $n++) { print " $A[$n]" }; 
        print "\n"; 
      }
    }
  ' | sort -k1 > $train_dir/transcripts.txt 

find $HKUST_TEXT_DIR -iname "*.txt" | grep -i "trans/dev" | xargs cat |\
  iconv -f GBK -t utf-8 - | perl -e '
    while (<STDIN>) {
      @A = split(" ", $_);
      if (@A <= 1) { next; }
      if ($A[0] eq "#") { $utt_id = $A[1]; } 
      if (@A >= 3) {
        $A[2] =~ s:^([AB])\:$:$1:; 
        printf "%s-%s-%06.0f-%06.0f", $utt_id, $A[2], 100*$A[0] + 0.5, 100*$A[1] + 0.5; 
        for($n = 3; $n < @A; $n++) { print " $A[$n]" }; 
        print "\n"; 
      }
    }
  ' | sort -k1  > $dev_dir/transcripts.txt



#transcripts normalization and segmentation 
#(this needs external tools),
#Download and configure segment tools   
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz 
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
  python setup.py build 
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi

cat $train_dir/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > $train_dir/text

cat $dev_dir/transcripts.txt |\
  sed -e 's/<foreign language=\"[a-zA-Z]\+\">/ /g' |\
  sed -e 's/<\/foreign>/ /g' |\
  sed -e 's/<noise>\(.\+\)<\/noise>/\1/g' |\
  sed -e 's/((\([^)]\{0,\}\)))/\1/g' |\
  local/hkust_normalize.pl |\
  python local/hkust_segment.py |\
  awk '{if (NF > 1) print $0;}' > $dev_dir/text

# some data is corrupted. Delete them
cat $train_dir/text | grep -v 20040527_210939_A901153_B901154-A-035691-035691 | egrep -v "A:|B:" > tmp
mv tmp $train_dir/text

#Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#sw02001-A_000098-001156 sw02001-A 0.98 11.56


awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' <$train_dir/text > $train_dir/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' $train_dir/sph.flist > $train_dir/sph.scp

awk '{ segment=$1; split(segment,S,"-"); side=S[2]; audioname=S[1];startf=S[3];endf=S[4];
   print segment " " audioname "-" side " " startf/100 " " endf/100}' <$dev_dir/text > $dev_dir/segments
awk '{name = $0; gsub(".sph$","",name); gsub(".*/","",name); print(name " " $0)}' $dev_dir/sph.flist > $dev_dir/sph.scp



sph2pipe=`cd ../../..; echo $PWD/tools/sph2pipe_v2.5/sph2pipe`
[ ! -f $sph2pipe ] && echo "Could not find the sph2pipe program at $sph2pipe" && exit 1;

cat $train_dir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
    printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
   sort > $train_dir/wav.scp || exit 1;

cat $dev_dir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); 
    printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
   sort > $dev_dir/wav.scp || exit 1;
#side A - channel 1, side B - channel 2

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.
cat $train_dir/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > $train_dir/reco2file_and_channel || exit 1;
cat $dev_dir/wav.scp | awk '{print $1}' | \
  perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_"; print "$1-$2 $1 $2\n"; ' \
  > $dev_dir/reco2file_and_channel || exit 1;


cat $train_dir/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > $train_dir/utt2spk || exit 1;
cat $train_dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $train_dir/spk2utt || exit 1;

cat $dev_dir/segments | awk '{spk=substr($1,1,33); print $1 " " spk}' > $dev_dir/utt2spk || exit 1;
cat $dev_dir/utt2spk | sort -k 2 | utils/utt2spk_to_spk2utt.pl > $dev_dir/spk2utt || exit 1;

echo HKUST data preparation succeeded
  
exit 1;
