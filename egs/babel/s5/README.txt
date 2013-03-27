How to setup the BABEL database training environment
====================================================
a) Preparation: you need to make sure the BABEL data and the F4DE scoring software
   is set up as it is in JHU, or change this setup accordingly.  This will probably 
   be hard and will involve some trial and error.  Some relevant pathnames can be 
   found in conf/lang/* and ./path.sh

   Link one of the config files in conf/languages to ./lang.conf.  E.g.:
    ln -s conf/languages/105-turkish-limitedLP.official.conf lang.conf
   

b) If you plan to work on one or more languages, the following approach is advised.
    aa) create empty directory somewhere according to your choice
    ab) symlink all the directories here to that directory
    ac) copy cmd.sh and path.sh (you will probably need to do some changes in these)
    ad) link the necessary scripts ( see below )
    ae) link the appropriate language-specific config file to lang.conf in
        each directory.

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


Official NIST submission preparation
==================================================
The make_release.sh script might come handy.
The scripts evaluates the performance of the sgmm2_mmi_b.0.1 system on 
the eval.uem dataset and chooses the same set of parameters to 
determine the path inside the test.uem dataset. 

./make_release.sh --relname defaultJHU --lp FullLP --lr BaseLR --ar NTAR  \
  conf/languages/106-tagalog-fullLP.official.conf /export/babel/data/releases



