
To train models, for an arbitrary mic, run something like:

./run_shared.sh

and then

./run.sh --mic mic_type 

where mic_type depends on whether you want to use individual headset mic (ihm),
 distant (but beamformed) multiple mics (mdm) or distant single mic (d1...d4, pda1, pda2).
Mutliple distant microphone (mdm) setup is using only 4 PDA mics. Look below
for more details, more on notations and a typical meeting layout ICSI followed.


About ICSCI corpora and this particular recipe
=================================================================================

This recipe builds ASR models using ICSI data, see [1] for a description, or [2]
to access the data. The correposning paper describing the ICSI corpora is [3]

[1] http://www1.icsi.berkeley.edu/Speech/mr/
[2] http://groups.inf.ed.ac.uk/ami/icsi/
[3] A Janin, D Baron, J Edwards, D Ellis, D Gelbart, N Morgan, B Peskin,
    T Pfau, E Shriberg, A Stolcke, and C Wooters, The ICSI meeting corpus. 
    in Proc IEEE ICASSP, 2003, pp. 364-367


ICSI data did not come with any pre-defined splits for train/valid/test sets as it was
mostly used as a training data for NIST RT evaluations. Some portions of unrelased ICSI 
data (as a part of corpora) can be found in, for example, NIST RT04 amd RT05 evaluation sets.

This recipe, however, to be self-contained factors out development and evaluation sets
in a way to minimise the speaker-overlap between different partitions. This is based on
[4] where dev uses {Bmr021 and Bns00} while eval uses {Bmr013, Bmr018 and Bro021}.

[4] S Renals and P Swietojanski, Neural networks for distant speech recognition. 
    in Proc IEEE HSCMA 2014 pp. 172-176. DOI:10.1109/HSCMA.2014.6843274


Below description is (mostly) copied from ICSI documentation for convenience.

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

The mapping from the above, to the actual waveform channels in the corpora is:
                                                                                
D1 - chanE                                                                      
D2 - chanF                                                                      
D3 - chan6                                                                      
D4 - chan7                                                                      
PDA left - chanC                                                                
PDA right - chanD 

