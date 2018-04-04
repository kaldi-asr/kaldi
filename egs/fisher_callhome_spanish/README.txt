Kaldi recipe for the Fisher and Callhome Spanish Corpora

About the Fisher Spanish Corpus
  Fisher Spanish - Speech was developed by the Linguistic 
  Data Consortium (LDC) and consists of audio files covering
  roughly 163 hours of telephone speech from 136 native
  Caribbean Spanish and non-Caribbean Spanish speakers.
  Full orthographic transcripts of these audio files are available
  in LDC2010T04

  Speech : LDC2010S01
  Transcripts : LDC2010T04

About the Callhome Spanish Corpus
  The CALLHOME Spanish corpus of telephone speech consists
  of 120 unscripted telephone conversations between native speakers of Spanish.
  All calls, which lasted up to 30 minutes, originated in North America
  and were placed to international locations. Most participants called
  family members or close friends.

  Speech : LDC96S35
  Transcripts : LDC96T17

The LDC Spanish rule based lexicon
  The CALLHOME Spanish collection includes a lexical component. 
  The CALLHOME Spanish Lexicon consists of 45,582 words and contains
  separate information fields with phonological, morphological and
  frequency information for each word.

  Lexicon : LDC96L16


Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s5: This recipe is based on the WSJ s5 recipe. It works with the 
      the transcripts (available along with the script in LDC97T19). In addition, 
      it uses a phonetic lexicon generated using the rules based LDC lexicon. 
      The recipe follows the Triphone+SGMM+SAT+fMLLR+SGMM+DNN pipeline. It uses data
      partitions as specified by LDC in the Callhome corpus description. For Fisher
      custom partitions are available (check the run.sh file for the location 
      of the split file : This can be changed).



