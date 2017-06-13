#!/bin/bash
#
# Johns Hopkins University : (Gaurav Kumar)
# The input is the Callhome Egyptian Arabic Dataset which contains *.sph files
# In addition the transcripts are needed as well.

#TODO: Rewrite intro, copyright stuff and dir information
# To be run from one directory above this script.

stage=0

export LC_ALL=C


if [ $# -lt 6 ]; then
   echo "Arguments should be the location of the Callhome Egyptian Arabic Speech and Transcript Directories, se
e ../run.sh for example."
   exit 1;
fi

cdir=`pwd`
dir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
tmpdir=`pwd`/data/local/tmp

mkdir -p $dir
mkdir -p $tmpdir

. ./path.sh || exit 1; # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi
cd $dir

# Make directory of links to the ECA data.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/
ln -s $* links

# Basic spot checks to see if we got the data that we needed
if [ ! -d links/LDC97S45 -o ! -d links/LDC97T19 ];
then
        echo "The speech and the data directories need to be named LDC97S45 and LDC97T19 respectively"
        exit 1;
fi
if [ ! -d links/LDC2002S37 -o ! -d links/LDC2002T38 ];
then
        echo "The Callhome supplement directories need to be named LDC2002S37 and LDC2002T38."
        o
        exit 1;
fi
if [ ! -d links/LDC2002S22 -o ! -d links/LDC2002T39 ];
then
        echo "The H5-ECA directories need to be named LDC2002S22 and LDC2002T39."
        exit 1;
fi

if [ ! -d links/LDC97S45/CALLHOME/ARABIC/DEVTEST -o ! -d links/LDC97S45/CALLHOME/ARABIC/EVLTEST -o ! -d links/LDC97S45/CALLHOME/ARABIC/TRAIN ];
then
        echo "Dev, Eval or Train directories missing or not properly organised within the speech data dir"
        exit 1;
fi

#Check the transcripts directories as well to see if they exist
if [ ! -d links/LDC97T19/callhome_arabic_trans_970711/transcrp/devtest -o ! -d links/LDC97T19/callhome_arabic_trans_970711/transcrp/evaltest -o ! -d links/LDC97T19/callhome_arabic_trans_970711/transcrp/train ]
then
        echo "Transcript directories missing or not properly organised"
        exit 1;
fi

if [ ! -d links/LDC2002S37/SPEECH ];
then
        echo "Callhome supplement directories missing or not properly organised within the speech data dir"
        exit 1;
fi

if [ ! -d links/LDC2002T38/ch_ara_transcr_suppl/transcr ]
then
        echo "Callhome supplement Transcript directories missing or not properly organised"
        exit 1;
fi

if [ ! -d links/LDC2002S22/SPEECH ];
then
        echo "H5 directories missing or not properly organised within the speech data dir"
        exit 1;
fi

if [ ! -d links/LDC2002T39/transcr ]
then
        echo "H5 Transcript directories missing or not properly organised"
        exit 1;
fi

speech_train=$dir/links/LDC97S45/CALLHOME/ARABIC/TRAIN
speech_dev=$dir/links/LDC97S45/CALLHOME/ARABIC/DEVTEST
speech_test=$dir/links/LDC97S45/CALLHOME/ARABIC/EVLTEST
transcripts_train=$dir/links/LDC97T19/callhome_arabic_trans_970711/transcrp/train/roman
transcripts_dev=$dir/links/LDC97T19/callhome_arabic_trans_970711/transcrp/devtest/roman
transcripts_test=$dir/links/LDC97T19/callhome_arabic_trans_970711/transcrp/evaltest/roman
speech_sup=$dir/links/LDC2002S37/SPEECH
transcripts_sup=$dir/links/LDC2002T38/ch_ara_transcr_suppl/transcr
speech_h5=$dir/links/LDC2002S22/SPEECH
transcripts_h5=$dir/links/LDC2002T39/transcr

fcount_train=`find ${speech_train} -iname '*.SPH' | wc -l`
fcount_dev=`find ${speech_dev} -iname '*.SPH' | wc -l`
fcount_test=`find ${speech_test} -iname '*.SPH' | wc -l`
fcount_t_train=`find ${transcripts_train} -iname '*.txt' | wc -l`
fcount_t_dev=`find ${transcripts_dev} -iname '*.txt' | wc -l`
fcount_t_test=`find ${transcripts_test} -iname '*.txt' | wc -l`
fcount_sup=`find ${speech_sup} -iname '*.SPH' | wc -l`
fcount_t_sup=`find ${transcripts_sup} -iname '*.txt' | wc -l`
fcount_h5=`find ${speech_h5} -iname '*.SPH' | wc -l`
fcount_t_h5=`find ${transcripts_h5} -iname '*.txt' | wc -l`

#Now check if we got all the files that we needed
if [ $fcount_train != 80 -o $fcount_dev != 20 -o $fcount_test != 20 -o $fcount_t_train != 80 -o $fcount_t_dev != 20 -o $fcount_t_test != 20 ];
then
        echo "Incorrect number of files in the data directories"
        echo "The paritions should contain 80/20/20 files"
        exit 1;
fi
if [ $fcount_sup != 20 -o $fcount_t_sup != 20 ];
then
        echo "Incorrect number of files in the ECA sup data directories"
        echo "The paritions should contain 20/20 files"
        exit 1;
fi
if [ $fcount_h5 != 20 -o $fcount_t_h5 != 20 ];
then
        echo "Incorrect number of files in the H5 data directories"
        echo "The paritions should contain 20/20 files"
        exit 1;
fi

if [ $stage -le 0 ]; then
	#Gather all the speech files together to create a file list
	(
	    find $speech_train -iname '*.sph';
	    find $speech_dev -iname '*.sph';
	    find $speech_test -iname '*.sph';
      find $speech_sup -iname '*.sph';
      find $speech_h5 -iname '*.sph';
	)  > $tmpdir/callhome_train_sph.flist

	#Get all the transcripts in one place

  (
    find $transcripts_train -iname '*.txt';
    find $transcripts_dev -iname '*.txt';
    find $transcripts_test -iname '*.txt';
    find $transcripts_sup -iname '*.txt';
    find $transcripts_h5 -iname '*.txt';
  )  > $tmpdir/callhome_train_transcripts.flist

fi

if [ $stage -le 1 ]; then
	$local/callhome_make_trans.pl $tmpdir
	mkdir -p $dir/train_all
	mv $tmpdir/reco2file_and_channel $dir/train_all/
fi

if [ $stage -le 2 ]; then
  sort $tmpdir/text.1 | grep -v '((' | \
  awk '{if (NF > 1){ print; }}' | \
  sed 's:<\s*[/]*\s*\s*for[ei][ei]g[nh]\s*\w*>::g' | \
  sed 's:<lname>\([^<]*\)<\/lname>:\1:g' | \
  sed 's:<lname[\/]*>::g' | \
  sed 's:<laugh>[^<]*<\/laugh>:[laughter]:g' | \
  sed 's:<\s*cough[\/]*>:[noise]:g' | \
  sed 's:<sneeze[\/]*>:[noise]:g' | \
  sed 's:<breath[\/]*>:[noise]:g' | \
  sed 's:<lipsmack[\/]*>:[noise]:g' | \
  sed 's:<background>[^<]*<\/background>:[noise]:g' | \
  sed -r 's:<[/]?background[/]?>:[noise]:g' | \
  #One more time to take care of nested stuff
  sed 's:<laugh>[^<]*<\/laugh>:[laughter]:g' | \
  sed -r 's:<[/]?laugh[/]?>:[laughter]:g' | \
  #now handle the exceptions, find a cleaner way to do this?
  sed 's:<foreign langenglish::g' | \
  sed 's:</foreign::g' | \
  sed -r 's:<[/]?foreing\s*\w*>::g' | \
  sed 's:</b::g' | \
  sed 's:<foreign langengullís>::g' | \
  sed 's:foreign>::g' | \
  sed 's:>::g' | \
  #How do you handle numbers?
  grep -v '()' | \
  #Now go after the non-printable characters
  sed -r 's:¿::g' > $tmpdir/text.2

  cp $tmpdir/text.2 $dir/train_all/text


  #Create segments file and utt2spk file
  ! cat $dir/train_all/text | perl -ane 'm:([^-]+)-([AB])-(\S+): || die "Bad line $_;"; print "$1-$2-$3 $1-$2\n"; ' > $dir/train_all/utt2spk \
  && echo "Error producing utt2spk file" && exit 1;

  # Remove utterances that have the same start and end time. Corresponding text entries will be removed when use
  # fix_data_dir.sh and validate_data_dir.sh later
  cat $dir/train_all/text | perl -ane 'm:((\S+-[AB])-(\d+)-(\d+))\s: || die; $utt = $1; $reco = $2;
 $s = sprintf("%.2f", 0.01*$3); $e = sprintf("%.2f", 0.01*$4); print "$utt $reco $s $e\n"; ' | \
   awk '{if (!(NF != 4 || ($4 <= $3 && $4 != -1))) {print $0}}' >$dir/train_all/segments

  $utils/utt2spk_to_spk2utt.pl <$dir/train_all/utt2spk > $dir/train_all/spk2utt
fi

if [ $stage -le 3 ]; then
  cat $tmpdir/callhome_train_sph.flist | perl -ane 'm:/([^/]+)\.SPH$: || die "bad line $_; ";  print lc($1)," $_"; ' > $tmpdir/sph.scp
  cat $tmpdir/sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
  sort -k1,1 -u  > $dir/train_all/wav.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  # Build the speaker to gender map, the temporary file with the speaker in gender information is already created by fsp_make_trans.pl.
  cd $cdir
  #TODO: needs to be rewritten
  $local/callhome_make_spk2gender > $dir/train_all/spk2gender
fi

echo "CALLHOME ECA Data preparation succeeded."
