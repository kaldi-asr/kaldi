About the Fisher-English corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  The data is similar to Switchboard but the transcription was mostly
    done in a "faster", lower-quality way.

    Fisher comes in two parts, and the text and speech have separate LDC numbers.
    This recipe uses both parts.  The LDC numbers are

    The speech: LDC2004S13, LDC2005S13
    The text: LDC2004T19, LDC2005T19
 

Each subdirectory of this directory contains the
scripts for a sequence of experiments.

  s5: This recipe is being worked on, it has the initial stages of
      training ready.  Note that the data normalization is not compatible
      with our Switchboard setup, we have retained the conventions
      of the Fisher corpus, e.g. lower-case, and acronyms like c._n._n.



