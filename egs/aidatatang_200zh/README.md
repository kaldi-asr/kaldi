Aidatatang_200zh is a free Chinese Mandarin speech corpus provided by Beijing DataTang Technology Co., Ltd under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License. 

**About the aidatatang_200zh corpus:**

- The corpus contains 200 hours of acoustic data, which is mostly mobile recorded data.
- 600 speakers from different accent areas in China are invited to participate in the recording.
- The transcription accuracy for each sentence is larger than 98%.
- Recordings are conducted in a quiet indoor environment. 
- The database is divided into training set, validation set, and testing set in a ratio of 7: 1: 2.
- Detail information such as speech data coding and speaker information is preserved in the metadata file.
- Segmented transcripts are also provided.

You can get the corpus from [here](https://www.datatang.com/webfront/opensource.html). 

DataTang is a community of creators-of world-changers and future-builders. We're invested in collaborating with a diverse set of voices in the AI world, and are excited about working on large-scale projects. Beyond speech, we're providing multiple resources in image, and text. For more details, please visit [datatang](<https://www.datatang.com/>).

**About the recipe:**

To demonstrate that this corpus is a reasonable data resource for Chinese Mandarin speech recognition research, a baseline recipe is provided here for everyone to explore their own systems easily and quickly.

In this directory, each subdirectory contains the scripts for a sequence of experiments. The recipe in subdirectory "s5" is based on the hkust s5 recipe and aishell s5 recipe. It generates an integrated phonetic lexicon with CMU dictionary and cedit dictionary. This recipe follows the Mono+Triphone+SAT+fMLLR+DNN pipeline. In addition, this directory will be extended as scripts for speaker diarization and so on are created.
