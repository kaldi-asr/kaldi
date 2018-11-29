Improved baseline for REVERB challenge based on Kaldi
==============================================================================

updated
Wed Nov 28 11:36:30 EST 2018 Szu-Jui Chen <schen146@jhu.edu>

updated
Wed Apr 29 19:10:33 EDT 2015 Shinji Watanabe <watanabe@merl.com>

updated 
Wed Apr  9 12:14:02 CEST 2014 Felix Weninger <felix@weninger.de>

original:
Wed Nov  6 14:47:59 EST 2013 Felix Weninger <felix@weninger.de>

Key specs:
- MFCC-LDA-STC front-end(not sure)
- TDNN acoustic model
- Utterance-based adaptation using basis fMLLR
- Tri-gram LM minimum Bayes risk decoding(not sure)

RESULT:
For experiment results, please see RESULTS for more detail

REFERENCE:
++++++++
If you find this software useful for your own research, please cite the
following paper:

Felix Weninger, Shinji Watanabe, Jonathan Le Roux, John R. Hershey, Yuuki
Tachioka, Jürgen Geiger, Björn Schuller, Gerhard Rigoll: "The MERL/MELCO/TUM
system for the REVERB Challenge using Deep Recurrent Neural Network Feature
Enhancement", Proc. REVERB Workshop, IEEE, Florence, Italy, May 2014.


INSTRUCTIONS:
+++++++++++++
1) Execute the training and recognition steps by

   ./run.sh

   Depending on your system specs (# of CPUs, RAM) you might want (or have) to 
   change the number of parallel jobs -- this is controlled by the nj
   and decode_nj variables (# of jobs for training, for decoding).

