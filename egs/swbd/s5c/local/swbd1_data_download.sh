#!/bin/bash

# Switchboard-1 training data preparation customized for Edinburgh
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

## The second input is optional, which should point to a directory containing
## Switchboard transcriptions/documentations (specifically, the conv.tab file).
## If specified, the script will try to use the actual speaker PINs provided 
## with the corpus instead of the conversation side ID (Kaldi default). We 
## will be using "find" to locate this file so we don't make any assumptions
## on the directory structure. (Peng Qi, Aug 2014)

. path.sh

#check existing directories
if [ $# != 1 -a $# != 2 ]; then
  echo "Usage: swbd1_data_prep_edin.sh /path/to/SWBD [/path/to/SWBD_DOC]"
  exit 1; 
fi 

SWBD_DIR=$1

dir=data/local/train
mkdir -p $dir


# Audio data directory check
if [ ! -d $SWBD_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
[ ! -x $sph2pipe ] \
  && echo "Could not execute the sph2pipe program at $sph2pipe" && exit 1;


# Trans directory check
if [ ! -d $SWBD_DIR/transcriptions/swb_ms98_transcriptions ]; then
  ( 
    cd $dir;
    if [ ! -d swb_ms98_transcriptions ]; then
      echo " *** Downloading trascriptions and dictionary ***" 
      wget http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz ||
      wget http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
      tar -xf switchboard_word_alignments.tar.gz
    fi
  )
else
  echo "Directory with transcriptions exists, skipping downloading"
  [ -f $dir/swb_ms98_transcriptions ] \
    || ln -sf $SWBD_DIR/transcriptions/swb_ms98_transcriptions $dir/
fi
