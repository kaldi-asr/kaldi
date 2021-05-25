National Speech Corpus

Link: https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

The National Speech Corpus (NSC) is a locally accented and contextualized repository of audio, accompanying transcripts and lexicon in the English Language.

It is being released as part of an effort to improve the accuracy of Automatic Speech Recognition (ASR) technologies for Singapore.

Original ABOUT.txt contents:

The entire corpus is organised into a few parts. 

Part 1 features about 1000 hours of prompted recordings of phonetically-balanced scripts from about 1000 local English speakers. 

Part 2 presents about 1000 hours of prompted recordings of sentences randomly generated from words based on people, food, location, brands, etc, from about 1000 local English speakers as well. Transcriptions of the recordings have been done orthographically and are available for download.

Part 3 consists of about 1000 hours of conversational data recorded from about 1000 local English speakers, split into pairs. The data includes conversations covering daily life and of speakers playing games provided. 

Parts 1 and 2 were recorded in quiet rooms using 3 microphones: a headset/ standing microphone (channel 0), a boundary microphone (channel 1), and a mobile phone (channel 3). Recordings that are available for download here have been down-sampled to 16kHz. Details of the microphone models used for each speaker as well as some corresponding non-personal and anonymized information can be found in the accompanying spreadsheets. 

Part 3's recordings were split into 2 environments. In the Same Room environment where speakers were in same room, the recordings were done using 2 microphones: a close-talk mic and a boundary mic. In the Separate Room environment, speakers were separated into individual rooms. The recordings were done using 2 microphones in each room: a standing mic and a telephone. 



Directory Structure

The root folder contains subdirectories for Part 1, Part 2 and Part 3, a LEXICON subdirectory, a LICENCE file and a VERSION file. 

Part 1 and Part 2’s subdirectories are further organized into a DATA subdirectory, DOC folder for speaker and microphone information and transcription documentation, as well as a PROMPT INFO file. The DATA directory contains the audio recordings and corresponding transcriptions organized by the channel. The structure is as follows: 
•	/SCRIPT: containing orthographic transcripts saved in TXT format and organized by speaker number and session number (0 or 1)
•	/WAVE: containing audio files in WAV format, sampled at 16kHz, zipped into folders by the speaker number. Each folder further separates the audio files by the session number.

Part 3 is further organised into a six subdirectories, 3 for each recording environment (Same Room or Separate Room). Among each group of 3 subdirectories, 1 contains transcriptions, while the remaining 2 contain audio data from each of the two microphones used for the environment. There is also a manifest document at the root of the Part 3 folder that lists the files released.

Summary of Part 3 data organization:
- Same Room environment, files organized by speaker number:
    /Scripts Same: Orthographic transcripts saved in TextGrid format
    /Audio Same BoundaryMic: Audio files in WAV format recorded using the boundary mic, sampled at 16kHz
    /Audio Same CloseMic: Audio files in WAV format recorded using the close-talk mic, sampled at 16kHz

- Separate Room environment, files organized by speaker number and session number:
    /Scripts Separate: Orthographic transcripts saved in TextGrid format 
    /Audio Separate IVR: Audio files in WAV format recorded using the telephone, sampled at 16kHz
    /Audio Separate StandingMic: Audio files in WAV format recorded using the standing mic, sampled at 16kHz

The LEXICON subdirectory contains 2 files:
- Phonetic Inventory: containing a description of the identified phones of Singapore English
- LEXICON.txt: containing the phonetic transcription of all valid words in Part 1, Part 2 and Part 3


Updates

Updates will occur to this folder directly and at any time:
- The version file will be changed to reflect this;
- Depending on your Dropbox settings, your directory may SYNC the changes automatically;
- UPDATE HISTORY.txt will give a brief description or summary of the update;
- Older versions of the corpus will not be available.

