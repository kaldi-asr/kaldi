 This directory (lre07) contains example recipes for the 2007 NIST
 Language Evaluation.  The subdirectory v1 demonstrates our single best LID
 system, which is an I-Vector based recipe.  Different LID systems combine
 easily to improve accuracy; contained in the subdirectory v2 is a fusion
 of the I-Vector-based system in v1 with an additional I-Vector-based system
 that using MFCC+Pitch features.

 The following LDC corpora are used during training:
    
   SRE 2008 training set:                LDC2011S05 
   CALLFRIEND Vietnamese:                LDC96S60     
   CALLFRIEND Tamil:                     LDC96S59
   CALLFRIEND Japanese:                  LDC96S53
   CALLFRIEND Hindi:                     LDC96S52
   CALLFRIEND German:                    LDC96S51
   CALLFRIEND Farsi:                     LDC96S50
   CALLFRIEND French:                    LDC96S48
   CALLFRIEND Standard Arabic:           LDC96S49
   CALLFRIEND Korean:                    LDC96S54
   CALLFRIEND Mainland Chinese Mandarin: LDC96S55
   CALLFRIEND Taiwan Chinese Mandarin:   LDC96S56
   CALLFRIEND Caribbean Spanish:         LDC96S57
   CALLFRIEND Non-Caribbean Spanish:     LDC96S58
   LRE 1996:                             LDC?
   LRE 2003:                             LDC2006S31
   LRE 2005:                             LDC2008S05
   LRE 2007 Training Set:                LDC2009S05
   LRE 2009:                             LDC2014S06
 
 Note that some of the corpora, e.g., SRE 2008 and the LREs used for 
 training contain multiple languages.  Because of this, it isn't
 necessarily vital that all of the corpora are present in your system. 
 
 The NIST 2007 Language Evaluation (LDC2009S04) is used for testing. 
 
 This list will be updated as scripts for system development and testing
 (which will require additional data sources) are created. 

 The subdirectories "v1" and so on are different versions of the recipe.

