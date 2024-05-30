 This example demonstrates a traditional diarization system using a sliding-
 window iVector with PLDA scoring and agglomerative hierarchical clustering.

 The recipe uses the following data for system development.  This is in
 addition to the NIST SRE 2000 dataset used for evaluation (see ../README.txt).

     Corpus              LDC Catalog No.
     SRE2004             LDC2006S44
     SRE2005 Train       LDC2011S01
     SRE2005 Test        LDC2011S04
     SRE2006 Train       LDC2011S09
     SRE2006 Test 1      LDC2011S10
     SRE2006 Test 2      LDC2012S01
     SRE2008 Train       LDC2011S05
     SRE2008 Test        LDC2011S08
     SWBD2 Phase 2       LDC99S79
     SWBD2 Phase 3       LDC2002S06
     SWBD Cellular 1     LDC2001S13
     SWBD Cellular 2     LDC2004S07
