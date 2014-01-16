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

Each subdirectory of this directory contains the scripts for a sequence
of experiments.

  s3: Monophone GMM/HMM system trained with Maximum likelihood. Training
      is done with 61 phonemes, that are collapsed down to 39 phoneme
      during testing. Implemented by Navdeep Jaitly (ndjaitly@cs.toronto.edu)
      [from Dan: I believe this is now somewhat out of date, please us s5/]

  s4: Monophone, Triphone GMM/HMM systems trained with Maximum Likelihood.
      Training is done on 48 phonemes  (see- Lee and Hon: Speaker-Independent
      Phone Recognition Using Hidden Markov Models. IEEE TRANSACTIONS ON
      ACOUSTICS. SPEECH, AND SIGNAL PROCESSING, VOL. 31. NO. 11, PG. 1641-48,
      NOVEMBER 1989, ). Implemented by Arnab Ghoshal (arnab13@gmail.com)

  s5: the currently recommended recipe.
