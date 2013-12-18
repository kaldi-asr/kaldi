
 This directory (sre08) contains example scripts for speaker identification, not
 speech recognition.  The following LDC corpora are required to for the testing:
    
   SRE 2008 training set:  LDC2011S05 
   SRE 2008 test set:      LDC2011S08

 (but all the subdirectories require additional data for system development, see
 the corresponding README.txt files in subdirectories).

 Note:
 In the speaker id community the words "train", "test" and "development"
 are used in a different sense from in the speech recognition community.  In
 speaker-id land, the "development" data is the data you actually use to build
 the system; the "training" and "test" data are really both used to test how
 well the systme performs.  The "training" data is used as enrolment data for a
 particular speaker, and the evaluation consists of taking utterances from the
 "test" data and having the system decide whether they were uttered by various
 training speakers.


 The subdirectories "v1" and so on are different versions of the recipe;
 we don't call them "s1" etc., because they don't really correspond to
 the speech recognition recipes.


