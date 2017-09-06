 This recipe replaces the standard MFCC trained GMM of the v1 recipe with a
 tandem feature trained GMM. A time-delay deep neural network (TDNN) acoustic 
 model is used to extract the frame level phoneme posterior probabilities.
 After log, PCA, the resulted low dimensional features are fused with MFCC at 
 the feature level to get hybrid tandem feature. The recipe is based on 
 http://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_1120.pdf.
 See run.sh for updated results.

 The following describes data required for system development (on top of the 
 data for testing described in ../README.txt).  We use SWBD and the older 
 (prior to 2010) SREs to train the GMM and iVector extractor. To create an 
 in-domain system, the SREs are needed to train the PLDA backend. The TDNN is
 trained on Fisher English.
 
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
