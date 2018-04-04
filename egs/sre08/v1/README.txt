 Data required for system development (on top of the data for testing described
 in ../README.txt), consists of Fisher, past NIST SREs, and Switchboard
 cellular.  You can probably get by OK with just one part of Fisher.
  
                      Speech       Transcripts (see note)
   Fisher part 1     LDC2004S13        LDC2004T19
   Fisher part 2     LDC2005S13        LDC2005T19
   SRE 2004 Test     LDC2006S44
   SRE 2005 Test     LDC2011S04
   SWBD Cellular 1   LDC2001S13
   SWBD Cellular 2   LDC2004S07


Note:
 The distributions with the transcripts are not really needed for the
 transcripts themselves, but because that's where the speaker information
 resides (so we know which recordings are from the same speaker).  This is
 needed for PLDA estimation.  However, bear in mind that Fisher is not believed
 to be very good for things like PLDA estimation. In newer recipes such as
 ../../sre10/v1 we use past SRE data for PLDA estimation.
