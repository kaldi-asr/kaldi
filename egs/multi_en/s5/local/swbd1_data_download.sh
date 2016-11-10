#!/bin/bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/swbd1_data_download.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Specified path to path.sh
#  - Modified paths to match multi_en naming conventions
###########################################################################################

# Switchboard-1 training data preparation customized for Edinburgh
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

. ./path.sh

#check existing directories
if [ $# != 1 ]; then
  echo "Usage: swbd1_data_download.sh /path/to/SWBD"
  exit 1; 
fi 

SWBD_DIR=$1

dir=data/local/swbd
mkdir -p $dir

# Audio data directory check
if [ ! -d $SWBD_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1; 
fi  

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
