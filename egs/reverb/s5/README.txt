Improved baseline for REVERB challenge 
======================================

This is an improvement over "Improved multi condition training baseline" from Felix Weninger & Shinji Watanabe

Key specs:
- Nara-WPE and BeamformIt front-end enhancement
- TDNN acoustic model

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
