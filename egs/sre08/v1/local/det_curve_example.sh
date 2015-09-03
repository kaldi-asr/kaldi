#!/bin/bash

# This shell-script is just to show how you would plot det curves
# using DETWare.

# This assumes you have a file called "foo" with the scores in.

# Note: there are some comments at the bottom showing what you
# have to run in matlab.  It won't get run if you just run the script,
# you have to do it manually.

trials=data/sre08_trials/short2-short3-female.trials

for condition in $(seq 8); do
(
  # see http://www.itl.nist.gov/iad/mig/tests/sre/2008/official_results/
  # for interpretation of condition here,  e.g. 1 = "Interview train and test".
  # Condition 6 is "telephone train and test".
  # The EER is 
  awk '{print $3}' foo | paste - $trials | awk -v c=$condition '{n=4+c; if ($n == "Y") print $1, $4}' | grep -w target | \
    awk 'BEGIN {printf( "target = [ " );} {print $1} END{printf("];\n");}'
  awk '{print $3}' foo | paste - $trials | awk -v c=$condition '{n=4+c; if ($n == "Y") print $1, $4}' | grep -w nontarget | \
    awk 'BEGIN {printf( "nontarget = [ " );} {print $1} END{printf("];\n");}'
 ) > ~/DETware_v2.1/scores${condition}.m
done

# Note: the DETware_v2.1 directory is as extracted from the DETware package,
# which I got at.
# http://www.itl.nist.gov/iad/mig//tools/DETware_v2.1.targz.htm
# cd ~/DETware_v2.1/
# matlab
# and run at the matlab prompt:
# >> scores6
# >> [P_miss,P_fa] = Compute_DET(target, nontarget)
# >> Plot_DET(P_miss, P_fa, 'r')
# that particular result can be compared with the short2-short3 results here
# http://www.itl.nist.gov/iad/mig/tests/sre/2008/official_results/
# particularly the telephone-train, telephone-test condition which is here:
# http://www.itl.nist.gov/iad/mig/tests/sre/2008/official_results/dets/short2-short3.allPrimarySytems.16.det.png
# Note: this is the only condition we seem to be doing comparable to the systems there,
# presumably because it matches the training data we used.
