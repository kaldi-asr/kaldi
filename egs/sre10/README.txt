
 This directory (sre10) contains example scripts for the NIST SRE 2010
 speaker recognition evaluation. The following corpora are required:
    
   NIST SRE 2010 training set
   NIST SRE 2010 test set
 
 More details on NIST SRE 2010 can be found at the url
 http://www.itl.nist.gov/iad/mig/tests/sre/2010/. Additional data sources
 are required by the subdirectories. See the corresponding README.txt files 
 in the subdirectories for more details.

 The subdirectories "v1" and so on are different iVector-based speaker 
 recognition recipes. The recipe in v1 demonstrates a standard approach 
 using a full-covariance GMM-UBM, iVectors, and a PLDA backend. The example 
 in v2 replaces the GMM of the v1 recipe with a time-delay deep neural 
 network.

