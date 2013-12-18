


 Data required for system development (on top of the data for testing described
 in ../README.txt), is Fisher parts 1 and 2 (you could probably get by OK with
 just one of these).
  
                      Speech       Transcripts (see note)
   Fisher part 1     LDC2004S13        LDC2004T19
   Fisher part 2     LDC2005S13        LDC2005T19


Note:
 The distributions with the transcripts are not really needed for the
 transcripts themselves, but because that's where the speaker information
 resides (so we know which recordings are from the same speaker).  This is
 needed for PLDA estimation.  However, bear in mind that Fisher is not believed
 to be very good for things like PLDA estimation, so in future we may make a
 recipe where this is not required, and instead use past SRE data for
 things like PLDA estimation.

