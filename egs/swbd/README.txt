About the Switchboard corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz


Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s3: 
   This Switchboard recipe is not fully state-of-the-art, mostly because
   the language model and dictionary do not use any external sources of
   data, just what is in the LDC corpus.  As a result there is a
   high-perplexity language model and small dictionary.

  s5:
   This is the "new-new-style" recipe.  It's currently (late April 2012)
   being worked on actively and may change substantially.


