#!/bin/bash
#
# Copyright 2014  Gaurav Kumar.   Apache 2.0
# The input is the Callhome Spanish Dataset. (*.sph files) 
# In addition the transcripts are needed as well. 
# To be run from one directory above this script.

# Note: when creating your own data preparation scripts, it's a good idea
# to make sure that the speaker id (if present) is a prefix of the utterance
# id, that the output scp file is sorted on utterance id, and that the 
# transcription file is exactly the same length as the scp file and is also
# sorted on utterance id (missing transcriptions should be removed from the
# scp file using e.g. scripts/filter_scp.pl)

stage=0

export LC_ALL=C


if [ $# -lt 2 ]; then
   echo "Arguments should be the location of the Callhome Spanish Speech and Transcript Directories, se
e ../run.sh for example."
   exit 1;
fi

cdir=`pwd`
dir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils
tmpdir=`pwd`/data/local/tmp

. ./path.sh || exit 1; # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi
cd $dir

# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
#rm -r links/ 2>/dev/null
mkdir -p links/
ln -s $* links

# Basic spot checks to see if we got the data that we needed
if [ ! -d links/LDC96S35 -o ! -d links/LDC96T17 ];
then
        echo "The speech and the data directories need to be named LDC96S35 and LDC96T17 respecti
vely"
        exit 1;
fi

if [ ! -d links/LDC96S35/CALLHOME/SPANISH/SPEECH/DEVTEST -o ! -d links/LDC96S35/CALLHOME/SPANISH/SPEECH/EVLTEST -o ! -d links/LDC96S35/CALLHOME/SPANISH/SPEECH/TRAIN ];
then
        echo "Dev, Eval or Train directories missing or not properly organised within the speech data dir"
        exit 1;
fi

#Check the transcripts directories as well to see if they exist
if [ ! -d links/LDC96T17/callhome_spanish_trans_970711/transcrp/devtest -o ! -d links/LDC96T17/callhome_spanish_trans_970711/transcrp/evltest -o ! -d links/LDC96T17/callhome_spanish_trans_970711/transcrp/train ]
then
        echo "Transcript directories missing or not properly organised"
        exit 1;
fi

speech_train=$dir/links/LDC96S35/CALLHOME/SPANISH/SPEECH/TRAIN
speech_dev=$dir/links/LDC96S35/CALLHOME/SPANISH/SPEECH/DEVTEST
speech_test=$dir/links/LDC96S35/CALLHOME/SPANISH/SPEECH/EVLTEST
transcripts_train=$dir/links/LDC96T17/callhome_spanish_trans_970711/transcrp/train 
transcripts_dev=$dir/links/LDC96T17/callhome_spanish_trans_970711/transcrp/devtest 
transcripts_test=$dir/links/LDC96T17/callhome_spanish_trans_970711/transcrp/evltest 
                                                                                   
fcount_train=`find ${speech_train} -iname '*.SPH' | wc -l` 
fcount_dev=`find ${speech_dev} -iname '*.SPH' | wc -l`                                             
fcount_test=`find ${speech_test} -iname '*.SPH' | wc -l`                                             
fcount_t_train=`find ${transcripts_train} -iname '*.txt' | wc -l` 
fcount_t_dev=`find ${transcripts_dev} -iname '*.txt' | wc -l` 
fcount_t_test=`find ${transcripts_test} -iname '*.txt' | wc -l` 

#Now check if we got all the files that we needed
if [ $fcount_train != 80 -o $fcount_dev != 20 -o $fcount_test != 20 -o $fcount_t_train != 80 -o $fcount_t_dev != 20 -o $fcount_t_test != 20 ];                 
then                                                                               
        echo "Incorrect number of files in the data directories"                   
        echo "The paritions should contain 80/20/20 files"
        exit 1;                                                                    
fi   

if [ $stage -le 0 ]; then
	#Gather all the speech files together to create a file list
	(
	    find $speech_train -iname '*.sph';
	    find $speech_dev -iname '*.sph';
	    find $speech_test -iname '*.sph';
	)  > $tmpdir/callhome_train_sph.flist

	#Get all the transcripts in one place

	(                                                                              
    find $transcripts_train -iname '*.txt';
    find $transcripts_dev -iname '*.txt';
    find $transcripts_test -iname '*.txt';
    )  > $tmpdir/callhome_train_transcripts.flist 

fi

if [ $stage -le 1 ]; then
	$local/callhome_make_trans.pl $tmpdir
	mkdir -p $dir/callhome_train_all
	mv $tmpdir/callhome_reco2file_and_channel $dir/callhome_train_all/
fi

if [ $stage -le 2 ]; then                                                        
  sort $tmpdir/callhome.text.1 | grep -v '((' | \
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
  sed -r 's:¿::g' > $tmpdir/callhome.text.2

  CHARS=$(python -c 'print u"\u00BF\u00A1".encode("utf8")')
  sed -i 's/['"$CHARS"']//g' $tmpdir/callhome.text.2

  cp $tmpdir/callhome.text.2 $dir/callhome_train_all/callhome.text


  #Create segments file and utt2spk file
  ! cat $dir/callhome_train_all/callhome.text | perl -ane 'm:([^-]+)-([AB])-(\S+): || die "Bad line $_;"; print "$1-$2-$3 $1-$2\n"; ' > $dir/callhome_train_all/callhome_utt2spk \
  && echo "Error producing utt2spk file" && exit 1;

  cat $dir/callhome_train_all/callhome.text | perl -ane 'm:((\S+-[AB])-(\d+)-(\d+))\s: || die; $utt = $1; $reco = $2;
 $s = sprintf("%.2f", 0.01*$3); $e = sprintf("%.2f", 0.01*$4); print "$utt $reco $s $e\n"; ' >$dir/callhome_train_all/callhome_segments

  $utils/utt2spk_to_spk2utt.pl <$dir/callhome_train_all/callhome_utt2spk > $dir/callhome_train_all/callhome_spk2utt
fi

if [ $stage -le 3 ]; then
  for f in `cat $tmpdir/callhome_train_sph.flist`; do
    # convert to absolute path
    readlink -e $f 
  done > $tmpdir/callhome_train_sph_abs.flist

  cat $tmpdir/callhome_train_sph_abs.flist | perl -ane 'm:/([^/]+)\.SPH$: || die "bad line $_; ";  print lc($1)," $_"; ' > $tmpdir/callhome_sph.scp
  cat $tmpdir/callhome_sph.scp | awk -v sph2pipe=$sph2pipe '{printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2); printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);}' | \
  sort -k1,1 -u  > $dir/callhome_train_all/callhome_wav.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  # Build the speaker to gender map, the temporary file with the speaker in gender information is already created by fsp_make_trans.pl.
  cd $cdir
  #TODO: needs to be rewritten
  $local/callhome_make_spk2gender.sh > $dir/callhome_train_all/callhome_spk2gender
fi

# Rename files from the callhome directory
if [ $stage -le 5 ]; then
    cd $dir/callhome_train_all
    mv callhome.text text
    mv callhome_segments segments
    mv callhome_spk2utt spk2utt
    mv callhome_wav.scp wav.scp
    mv callhome_reco2file_and_channel reco2file_and_channel
    mv callhome_spk2gender spk2gender
    mv callhome_utt2spk utt2spk
fi

echo "CALLHOME spanish Data preparation succeeded."

exit 0;

