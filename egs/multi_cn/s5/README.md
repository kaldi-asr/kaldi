This is a Chinese speech recognition recipe that trains on all Chinese corpora on [OpenSLR](http://www.openslr.org), including:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)

This recipe was developed by Xingyu Na (Microsoft Corporation) and Hui Bu (AISHELL Foundation).

## Highlights

1. This recipe start from bootstraping small GMM models using small portion of data to speaker adaptive training using cleaned full partition, which is over 1k hours.
2. A general lexicon is prepared by combining CMU English dictionary and CC-CEDIT Chinese dictionary, then **expanded using G2P**.
3. A general language model is trained using all training transcriptions, while **corpus specific LMs** are optionally obtained by interpolated with the general LM.
4. Features are extracted in an online fashion.
5. A Chain model ready for **online ASR** is trained, prepared and evaluated.
6. Data preparation scripts are copied from existing recipes, so it is straightforward for any user to **expand the corpora**.
