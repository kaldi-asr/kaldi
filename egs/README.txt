
This directory contains example scripts that demonstrate how to 
use Kaldi.  Each subdirectory corresponds to a corpus that we have
example scripts for.  Currently these are all corpora available from
the Linguistic Data Consortium (LDC).

Explanations of the corpora are below.
Note: the easiest examples to work with are rm/s3 and wsj/s3.

 wsj: The Wall Street Journal corpus.  This is a corpus of read
    sentences from the Wall Street Journal, recorded under clean conditions.
    The vocabulary is quite large. 
    Available from the LDC as either: [ catalog numbers LDC93S6A (WSJ0) and LDC94S13A (WSJ1) ]
    or: [ catalog numbers LDC93S6B (WSJ0) and LDC94S13B (WSJ1) ]
    The latter option is cheaper and includes only the Sennheiser
    microphone data (which is all we use in the example scripts).

 rm: Resource Management.  Clean speech in a medium-vocabulary task consisting
    of commands to a (presumably imaginary) computer system.
    Available from the LDC as catalog number LDC93S3A (it may be possible to
    get the same data using combinations of other catalog numbers, but this
    is the one we used).

 tidigits: The TI Digits database, available from the LDC (catalog number LDC93S10).
   This is one of the oldest speech databases; it consists of a bunch of speakers
   saying digit strings.  It's not considered a "real" task any more, but can be useful
   for demos, tutorials, and the like.

 yesno: This is a simple recipe with some data consisting of a single person
   saying the words "yes" and "no", that can be downloaded from the Kaldi website.
   It's a very easy task, but useful for checking that the scripts run, or if
   you don't yet have any of the LDC data.
   

Recipes in progress (these may be less polished than the ones above).

 swbd: Switchboard.  A fairly large amount of telephone speech (2-channel, 8kHz
    sampling rate).
    This directory is a work in progress.
  
 gp: GlobalPhone.  This is a multilingual speech corpus.

 timit: TIMIT, which is an old corpus of carefully read speech.  
    LDC corpous LDC93S1  

