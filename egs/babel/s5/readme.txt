How to setup the BABEL database training environment
====================================================
a) If you plan to work only one language, no preparation is necessary
b) If you plan to work on one or more languages, the following approach is advised.
    aa) create empty directory somewhere according to your choice
    ab) symlink all the directories here to that directory
    ac) copy cmd.sh and path.sh (you will probably need to do some changes in these)
    ad) link the necessary scripts ( see bellow )

Running the training scripts
===================================================
The training scripts expect one parameter: the language config
You can see default scripts in the directory conf/languages/*.conf
The scripts *official.conf are the configurations for NIST defined data 
splits and dictionaries. 
Running training using these configs you will obtain the systems
functionally equivalent to the JHU systems (they can perform a little differently
depending on the CPU, matrix library and compiler switches).


What all the files in the s5/ directory are good for?
==================================================
* run-limited.sh -- training the LimitedLP systems
* run.sh -- training the FullLP condition systems

* run-test-limited.sh
* run-test.sh -- decoding of the eval part of the BABEL corpora using 
    the CMU UEM database (which you probably do not have). Still, you
    can use these scripts if you create the segmentation differently,
    prepare the kaldi directory and comment out the command local/cmu_uem2kaldi_dir.sh

* run-eval.sh -- evaluation of the 10h devel set (the training uses only 
    2h subset for tunning purposes)

* make_release.sh -- script for export of the best results to be prepared
    for official NIST submission

