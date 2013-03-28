#!/bin/bash

# The last version of this script that had all the commands in it from
# when we ran it from the eval is revision 2278.
# Note: this setup is still slightly in flux and won't work properly
# until I guess April 7th or around then.

# This script now just contains a note of what order you run things in.
# You are supposed to make a directory named e.g. ../s5-tagalog-limited,
# link most of the directories to there (but not exp/ or data/),
# link the appropriate conf file to lang.conf, e.g. 
#  ln -s conf/lang/106-tagalog-limitedLP.official.conf  lang.conf
# and run in this order:
# run-1-main.sh
# run-2a-nnet.sh and run-2b-bnf.sh 
# (these two can be run in parallel, but run-2b-bnf.sh should be done on a machine
#  with a GPU)
# run-3b-bnf-system.sh
# run-4-test.sh
