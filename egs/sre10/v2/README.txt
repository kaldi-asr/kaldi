 This recipe replaces the standard unsupervised GMM of the v1 recipe with a 
 UBM that uses a time-delay deep neural network (TDNN).  Posteriors from the
 TDNN are used in conjunction with features extracted using a standard approach
 for speaker recognition, to create the sufficient statistics for i-vector
 extraction.  The recipe also demonstrates a lightweight alternative in which
 a supervised GMM is derived from the TDNN posteriors. The recipe is based on
 http://www.danielpovey.com/files/2015_asru_tdnn_ubm.pdf. See run.sh for 
 updated results.

 The following describes data required for system development (on top of the 
 data for testing described in ../README.txt).  We use SWBD and the older 
 (prior to 2010) SREs to train the supervised-GMM and iVector extractor. To 
 create an in-domain system, the SREs are needed to train the PLDA backend.
 The TDNN is trained on Fisher English.
 
     Corpus              LDC Catalog No.
     SWBD2 Phase 2       LDC99S79
     SWBD2 Phase 3       LDC2002S06
     SWBD Cellular 1     LDC2001S13
     SWBD Ceullar 2      LDC2004S07
     SRE2004             LDC2006S44
     SRE2005 Train       LDC2011S01
     SRE2005 Test        LDC2011S04
     SRE2006 Train       LDC2011S09
     SRE2006 Test 1      LDC2011S10
     SRE2006 Test 2      LDC2012S01
     SRE2008 Train       LDC2011S05
     SRE2008 Test        LDC2011S08
     Fisher speech       LDC2004S13, LDC2005S13 
     Fisher test         LDC2004T19, LDC2005T19        
