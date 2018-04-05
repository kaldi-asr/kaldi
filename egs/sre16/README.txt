
 This directory (sre16) contains example scripts for the NIST SRE 2016
 speaker recognition evaluation. The following corpora are required to
 perform the evaluation:
    
   NIST SRE 2016 enroll set
   NIST SRE 2016 test set
 
 More details on NIST SRE 2016 can be found at the url
 https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016.

 Additional data sources (mostly past NIST SREs, Switchboard, etc) are
 required to train the systems in the subdirectories. See the
 corresponding README.txt files in the subdirectories for more details.

 The subdirectories "v1" and so on are different speaker recognition
 recipes. The recipe in v1 demonstrates a standard approach using a
 full-covariance GMM-UBM, iVectors, and a PLDA backend.  The example 
 in v2 demonstrates DNN speaker embeddings with a PLDA backend.

