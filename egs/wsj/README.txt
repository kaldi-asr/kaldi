
About the Wall Street Journal corpus:
    This is a corpus of read
    sentences from the Wall Street Journal, recorded under clean conditions.
    The vocabulary is quite large.   About 80 hours of training data.
    Available from the LDC as either: [ catalog numbers LDC93S6A (WSJ0) and LDC94S13A (WSJ1) ]
    or: [ catalog numbers LDC93S6B (WSJ0) and LDC94S13B (WSJ1) ]
    The latter option is cheaper and includes only the Sennheiser
    microphone data (which is all we use in the example scripts).


Each subdirectory of this directory contains the
scripts for a sequence of experiments.  Note: s3 is the "default" set of
scripts at the moment.

  s1: This setup is experiments with GMM-based systems with various 
      Maximum Likelihood 
      techniques including global and speaker-specific transforms.
      See a parallel setup in ../rm/s1 

  s2: This setup is experiments with pure hybrid system as well 
      as with Tandem bottleneck feature system.

  s3: This is the "new-style" recipe.   We recommend to look here first.
      The recipe uses a subset of the algorithms in s1, but adds
      cepstral mean subtraction (cms), and also uses more flexible, 
      general scripts.

  s5: This is the newer-style recipe.  It should run, although the RESULTS
      file is not finalized.


