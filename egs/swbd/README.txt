About the Switchboard corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz

    We are using the eval2000 evaluation data.  The acoustics are LDC2002S09 and
    the text is LDC2002T43.


Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s3: 
   This an older, now-deprecated recipe.

  s5: This is the "new-new-style" recipe.  
    All further work will be on top of this style of recipe.

  s5b: This is a cleaned-up version of s5, based on Arnab's
    "edinburgh" recipe, but still somewhat under construction.
