#!/bin/bash
# Author:   Ondrej Platek,2013, Apache 2.0
# Created:  09:56:45 13/03/2013
# Modified: 09:56:45 13/03/2013

if [ ! -d "$DATA_ROOT" ]; then
  echo "You need to set \"DATA_ROOT\" variable in path.sh to point to the directory to host data"
  exit 1
fi

# Ask about REMOVING the exp and data directory
if [ "$(ls -A exp 2>/dev/null)" ]; then
    read -p "Directory 'exp' is NON EMPTY. Do you want it to be OVERWRITTEN y/n?"
    case $REPLY in
        [Yy]* ) echo 'Deleting exp directory'; rm -rf exp;;
        [Nn]* ) echo 'Keeping exp directory';;
        * ) echo 'Keeping exp directory and cancelling..'; exit 1;;
    esac
fi

if [ "$(ls -A data 2>/dev/null)" ]; then
    read -p "Directory 'data' is NON EMPTY. Do you want it to be OVERWRITTEN y/n?"
    case $REPLY in
        [Yy]* ) echo 'Deleting data directory'; rm -rf data;;
        [Nn]* ) echo 'Reusing DATA SPLIT, LM, MFCC. SEE THE SCRIPT!';
                echo 'REUSING DATA from previous experiment!' \
                    'Check that everyN is THE SAME' >> exp/conf/train_conf.sh ;;
        * ) echo 'Keeping the data directory and cancelling..'
            exit 1;;
    esac
fi

if [ "$(ls -A ${MFCC_DIR} 2>/dev/null)" ]; then
    read -p "Directory '${MFCC_DIR}' is NON EMPTY. Do you want it to be OVERWRITTEN y/n?"
    case $REPLY in
        [Yy]* ) echo "Echo deleting ${MFCC_DIR}"; rm -rf "${MFCC_DIR}";;
        [Nn]* ) echo "Echo reusing MFCC at ${MFCC_DIR}!";
                echo 'REUSING MFCC from previous experiment!' \
                    'Check that the settings are THE SAME!' >> exp/conf/mfcc.conf
            ;;
        * ) echo 'Keeping the data directory and cancelling..'; 
            exit 1;;
    esac
fi

# make sure that the directories exists
mkdir -p "$MFCC_DIR"
mkdir -p "exp"
mkdir -p "data"

# Copy the current settings to exp directory
cp -r conf exp
cp cmd.sh path.sh exp/conf
git log -1 > exp/conf/git_log_state.log
git diff > exp/conf/git_diff_state.log
