About the Switchboard corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz

    We are using the eval2000 a.k.a. hub5'00 evaluation data.  The acoustics are
    LDC2002S09 and the text is LDC2002T43.

    We are also using the RT'03 test set, available as LDC2007S10.  Note: not
    all parts of the recipe test with this.

About the Fisher corpus for language modeling

  We use Fisher English training speech transcripts for language modeling, if
  they are available. The catalog number for part 1 transcripts is LDC2004T19,
  and LDC2005T19 for part 2.

Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s5: This is slightly out of date, please see s5c

  s5b: This is (somewhat less) out of date, please see s5c

  s5c: This is the current recipe.
