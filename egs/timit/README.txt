About TIMIT:

  Available as LDC corpus LDC93S1, TIMIT is one of the original 
  clean speech databases. Description of catalog from LDC
  (http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC93S1):

  "The TIMIT corpus of read speech is designed to provide speech data 
   for acoustic-phonetic studies and for the development and evaluation
   of automatic speech recognition systems. TIMIT contains broadband
   recordings of 630 speakers of eight major dialects of American English,
   each reading ten phonetically rich sentences. The TIMIT corpus includes
   time-aligned orthographic, phonetic and word transcriptions as well as 
   a 16-bit, 16kHz speech waveform file for each utterance."

   Note: please do not use this TIMIT setup as a generic example of how to run
   Kaldi, as TIMIT has a very nonstandard structure.  Any of the other setups
   would be better for this purpose: e.g. librispeech/s5 is quite nice, and is
   free; yesno is very tiny and fast to run and is also free; and wsj/s5 has an
   unusually complete set of example scripts which may however be confusing.

Each subdirectory of this directory contains the scripts for a sequence
of experiments.

  s5: Monophone, Triphone GMM/HMM systems trained with Maximum Likelihood,
      followed by SGMM and DNN recipe.
      Training is done on 48 phonemes (see- Lee and Hon: Speaker-Independent
      Phone Recognition Using Hidden Markov Models. IEEE TRANSACTIONS ON
      ACOUSTICS. SPEECH, AND SIGNAL PROCESSING, VOL. 31. NO. 11, PG. 1641-48,
      NOVEMBER 1989, ). In scoring we map to 39 phonememes, as is usually 
      done in conference papers. 
      The earlier versions of TIMIT scripts were implemented by Navdeep Jaitly,
      Arnab Ghoshal. Current version was developed by Bagher BabaAli and is 
      maintained by Karel Vesely (vesis84@gmail.com).

