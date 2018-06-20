# AISHELL-2

AISHELL-2 is by far the largest free speech corpus available for Mandarin ASR research.
## 1. DATA
### training data
* 1000 hours of speech data (around 1 million utterances)
* 1991 speakers (845 male and 1146 female)
* clean recording environment(studio or quiet living room)
* read speech
* reading prompts from various domain: entertainment, finance, technology, sports, control command, place of interest etc.
* near field recording via 3 parallel channels(iOS, Android, Microphone).
* iOS data is free for non-commercial research and education use (e.g. universities and colleges)

### evaluation data:
Currently we release AISHELL2-2018A-EVAL, containing:
* dev: 2500 utterances from 5 speaker
* test: 5000 utterances from 10 speakers

you can download above evaluation set from:
http://www.aishelltech.com/aishell_eval

we may update and release other evaluation sets on the website later, targeting on different applications and senarios.

## 2. RECIPE
Based on Kaldi standard system, AISHELL-2 provides a self-contained Mandarin ASR recipe, with:
* a word segmentation module, which is a must-have component for Chinese ASR systems
* an open-sourced Mandarin lexicon(DaCiDian)
* a simplified GMM training recipe
* acoustic channel adaptation recipe(AM fine-tuning)

# CONTACT
AISHELL foundation is a non-profit online organization, with members from speech industry and research institutes.

We hope AISHELL-2 corpus and recipe could be beneficial to the entire speech community.

Depends on your location and internet speed, we distribute the corpus in two ways:
* hard-disk delivery
* cloud-disk downloading

To apply for AISHELL-2 corpus for free, you need to fill in a very simple application form, confirming that:
* university department / education institute info
* only for non-commercial research / education use

AISHELL-foundation covers all data distribution fees (including the corpus, hard-disk cost etc)

Data re-distribution inside your university department is OK for convenience. However, users are not supposed to re-distribute AISHELL-2 to other universities or education institutes.

To get the application form, or you come across any problem with the recipe, contact us via:

aishell.foundation@gmail.com
