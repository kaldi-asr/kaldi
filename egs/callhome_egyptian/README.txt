About the Callhome Egyptian Arabic Corpus

  The CALLHOME Egyptian Arabic corpus of telephone speech consists of 120 unscripted 
  telephone conversations between native speakers of Egyptian Colloquial Arabic (ECA), 
  the spoken variety of Arabic found in Egypt. The dialect of ECA that this 
  dictionary represents is Cairene Arabic.

  This recipe uses the speech and transcripts available through LDC. In addition, 
  an Egyptian arabic phonetic lexicon (available via LDC) is used to get word to 
  phoneme mappings for the vocabulary. This datasets are:

  Speech : LDC97S45
  Transcripts : LDC97T19
  Lexicon : LDC99L22


Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s5: This recipe is based on the WSJ s5 recipe. It works with the 
      romanized version of the transcripts (available along with the
      script in LDC97T19). In addition, it uses a phonetic lexicon. 
      The recipe follows the Triphone+SGMM+SAT+fMLLR pipeline. It uses data
      partitions as specified by LDC in the corpora description. 



