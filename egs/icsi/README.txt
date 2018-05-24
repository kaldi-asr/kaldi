
To train models, for an arbitrary mic, go to s5 run something like (after you
set ICSI_DIR and ICSI_TRANS variables in either of below scripts:

./run_prepare_shared.sh

and then

./run.sh --mic mic_type 

where mic_type depends on whether you want to use individual headset mic (ihm),
distant (but beamformed) multiple mics (mdm) or distant single mic (D1...D4, PDA1, PDA2).
Mutliple distant microphones (mdm) setup is using only up to 4 PZM mics. Look below
for more details, on notations and a typical meeting layout that ICSI recordings followed.

Allowed mic_types for now are (ihm, sdm[1..4], mdm[2...4]). sdm4 will use D2 as a single
distant mic (look below for details, and why), mdm4 will combine D1...D4 using beamformer, etc.

About ICSI corpora and this particular recipe
=================================================================================

This recipe builds ASR models using ICSI data, and where possible follows the AMI s5b recipe.
See [1] for a description, or [2,3] to access the data. The correposning paper 
describing the ICSI corpora is [4]

[1] http://www1.icsi.berkeley.edu/Speech/mr/
[2] http://groups.inf.ed.ac.uk/ami/icsi/
[3] LDC: LDC2004S02 for audio, and LDC2004T04 for transcripts
[4] A Janin, D Baron, J Edwards, D Ellis, D Gelbart, N Morgan, B Peskin,
    T Pfau, E Shriberg, A Stolcke, and C Wooters, The ICSI meeting corpus. 
    in Proc IEEE ICASSP, 2003, pp. 364-367


ICSI data did not come with any pre-defined splits for train/valid/eval sets as it was
mostly used as a training material for NIST RT evaluations. Some portions of unrelased ICSI 
data (as a part of this corpora) can be found in, for example, NIST RT04 amd RT05 evaluation sets.

This recipe, however, to be self-contained factors out development and evaluation sets
in a way to minimise the speaker-overlap between different partitions, and to avoid known issues
with available recordings during evaluation. This recipe follows [5] where dev and eval sets are 
making use of {Bmr021, Bns00} and {Bmr013, Bmr018, Bro021}, respectively.

[5] S Renals and P Swietojanski, Neural networks for distant speech recognition. 
    in Proc IEEE HSCMA 2014 pp. 172-176. DOI:10.1109/HSCMA.2014.6843274


Below description is (mostly) copied from ICSI documentation for convenience.
=================================================================================

Simple diagram of the seating arrangement in the ICSI meeting room.             
                                                                                
The ordering of seat numbers is as specified below, but their                   
alignment with microphones may not always be as precise as indicated            
here. Also, the seat number only indicates where the participant                
started the meeting. Since most of the microphones are wireless, they           
were able to move around.                                                       
                                                                                                                                                                
   Door                                                                         
                                                                                
                                                                                
          1         2            3           4                                  
     -----------------------------------------------------------------------    
     |                      |                       |                      |   S
     |                      |                       |                      |   c
     |                      |                       |                      |   r
    9|   D1        D2       |   D3  PDA     D4      |                      |   e
     |                      |                       |                      |   e
     |                      |                       |                      |   n
     |                      |                       |                      |    
     -----------------------------------------------------------------------    
          8         7            6           5                                  
                                                                                
                                                                                
                                                                                
D1, D2, D3, D4  - Desktop PZM microphones                                       
PDA - The mockup PDA with two cheap microphones                                 
                                                                                
The following are the TYPICAL channel assignments, although a handful           
of meetings (including Bmr003, Btr001, Btr002) differed in assignment.         

The mapping from the above, to the actual waveform channels in the corpora,
and (this recipe for a signle distant mic case) is:
                                                                                
D1 - chanE - (this recipe: sdm3)                                                                      
D2 - chanF - (this recipe: sdm4)                                                                     
D3 - chan6 - (this recipe: sdm1)                                                                     
D4 - chan7 - (this recipe: sdm2)                                                                     
PDA left - chanC                                                                
PDA right - chanD 

-----------
Note (Pawel): The mapping for headsets is being extracted from mrt files. 
In cases where IHM channels are missing for some speakers in some meetings, 
in this recipe we either back off to distant channel (typically D2, default)
or (optionally) skip this speaker's segments entirely from processing. 
This is not the case for eval set, where all the channels come with the 
expected recordings, and split is the same for all conditions (thus allowing 
for direct comparisons between IHM, SDM and MDM settings).

