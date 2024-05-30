 This recipe replaces iVectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "xvectors."  The recipe in local/nnet3/xvector/tuning/run_xvector_1a.sh is
 closesly based on the following paper:

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 The recipe uses the following data for system development.  This is in
 addition to the NIST SRE 2016 dataset used for evaluation (see ../README.txt).
 
     Corpus              LDC Catalog No.
     SWBD2 Phase 1       LDC98S75
     SWBD2 Phase 2       LDC99S79
     SWBD2 Phase 3       LDC2002S06
     SWBD Cellular 1     LDC2001S13
     SWBD Cellular 2     LDC2004S07
     SRE2004             LDC2006S44
     SRE2005 Train       LDC2011S01
     SRE2005 Test        LDC2011S04
     SRE2006 Train       LDC2011S09
     SRE2006 Test 1      LDC2011S10
     SRE2006 Test 2      LDC2012S01
     SRE2008 Train       LDC2011S05
     SRE2008 Test        LDC2011S08
     SRE2010 Eval        LDC2017S06
     Mixer 6             LDC2013S03

 The following datasets are used in data augmentation.

     MUSAN               http://www.openslr.org/17
     RIR_NOISES          http://www.openslr.org/28
