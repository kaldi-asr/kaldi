SUMMARY
-------
KALDI recipe based on voxforge KALDI recipe 
http://vpanayotov.blogspot.cz/2012/07/voxforge-scripts-for-kaldi.html .
Requires KALDI installation and Linux environment. (Tested on Ubuntu 10.04 and 12.10.)
Written in Bash an Python 2.7.3.

DESCRIPTION
-----------
 * Our scripts prepare the data to expected format in s5/data. 
 * Stores experiments in s5/exp
 * steps/ contains common scripts from wsj/s5/utils
 * utils/ cotains common scritps from wsj/s5/utils
 * local/ contains scripts for data preparation to prepare s5/data structure
 * conf/ contains a few configuration files for KALDI


Runnning experiments
--------------------
Before running the experiments check the following files:
 * `conf` directory contains different configuration related for the training
 * `path.sh` just set up path for running Kaldi binaries and path to data.
    You should also setup `njobs` according your computer capabalities.
 * `cmd.sh` set training commands e.g. for SGE grid.
 * If you set up everything right, just launch `run.sh` It will create `mfcc`, `data` and `exp` directories.
   If any of them exists, it will ask you if you want them to be overwritten.
 ```bash
 ./run.sh | tee mylog.log # I always store the output to the log
 ```
 * I wrote a stupid script for collecting results. It's really beta software. It may crash, but it works for me.
 ```bash
$ local/results.py exp # specify the experiment directory wait a while
exp             RT coef         WER             SER
_ri3b_fmmi_b    2.42235533333   (19.45, 13)     (44.67, 11)
tri2b_mpe       0.37968465      (20.83, 20)     (47.2, 14)
mono            0.9478559       (52.42, 15)     (77.33, 14)
tri3b_mmi       0.357894733333  (19.77, 16)     (46.0, 11)
tri1            0.6558491       (27.12, 18)     (57.33, 20)
...
... and other results in plaintex
...
==================
\begin{tabular}{cccc}
exp             & RT coef        & WER         & SER        \\
_ri3b_fmmi_b    & 2.42235533333  & (19.45, 13) & (44.67, 11)\\
tri2b_mpe       & 0.37968465     & (20.83, 20) & (47.2, 14) \\
mono            & 0.9478559      & (52.42, 15) & (77.33, 14)\\
tri3b_mmi       & 0.357894733333 & (19.77, 16) & (46.0, 11) \\
tri1            & 0.6558491      & (27.12, 18) & (57.33, 20)\\
...
... and the same results in TeX
...

 ```
