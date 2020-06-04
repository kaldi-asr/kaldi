
 The MobvoiHotwords dataset is a ~144-hour corpus of wake word corpus which is
 publicly availble on https://www.openslr.org/87

 For wake word data, wake word utterances contain either 'Hi xiaowen' or 'Nihao
 Wenwen' are collected. For each wake word, there are about 36k utterances. All
 wake word data is collected from 788 subjects, ages 3-65, with different
 distances from the smart speaker (1, 3 and 5 meters). Different noises
 (typical home environment noises like music and TV) with varying sound
 pressure levels are played in the background during the collection.

 The recipe is in v1/

 The E2E LF-MMI recipe does not require any prior alignments for training
 LF-MMI, making the alignment more flexible during training. It can be optionally
 followed by a regular LF-MMI training to further improve the performance.

