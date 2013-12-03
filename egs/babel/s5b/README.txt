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

You run the scripts in order, i.e.
 run-1-main.sh
 run-2a-nnet.sh and run-2-bnf.sh may be run in parallel, but run-2-bnf.sh should be
    run on a machine that has a GPU.
 run-3-bnf-system.sh trains an SGMM system on top of bottleneck features from run-2-bnf.sh
 run-4-test.sh is decoding with provided segmentation (we get this from CMU)
 run-5-anydecode.sh seems to be decoding with the segmentation provided 



Official NIST submission preparation
==================================================
The make_release.sh script might come handy.
The scripts evaluates the performance of the sgmm2_mmi_b.0.1 system on 
the eval.uem dataset and chooses the same set of parameters to 
determine the path inside the test.uem dataset. 

./make_release.sh --relname defaultJHU --lp FullLP --lr BaseLR --ar NTAR  \
  conf/languages/106-tagalog-fullLP.official.conf /export/babel/data/releases



