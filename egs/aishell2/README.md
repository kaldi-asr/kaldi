# AISHELL-2

AISHELL-2 is by far the largest open-source speech corpus available for Mandarin ASR research.
## 1. DATA
### training data
* 1000 hours of speech data (around 1 million utterances)
* 1991 speakers (845 male and 1146 female)
* clean recording environment(studio or quiet living room)
* read speech
* reading prompts from various domain: entertainment, finance, technology, sports, control command, place of interest etc.
* near field recording via 3 parallel channels(iOS, Android, Microphone).
* data from iOS channel is open-sourced for research community

### evaluation data:
Currently we open-sourced AISHELL2-2018A-EVAL, containing:
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
* 80-dim FBank without pitch as NN input feature
* slightly different data augmentation setup(tempo perturbation, max-volume perturbation)
* acoustic channel adaptation recipe(AM fine-tuning)
* a real-time streaming ASR demo via laptop microphone (tested on MacOS)

# CONTACT
AISHELL foundation is a non-profit online organization, with members from speech industry and research institutes.

We hope the corpus and recipes that we open-sourced could be beneficial to the entire speech community.

Due to the scale of AISHELL-2 (more than 100G data for the open-sourced iOS channel), we have decided to distribute AISHELL-2 via hard-disk delivery.

To apply for the AISHELL-2 disk, or you come across any problem with the recipe, contact us via:

aishell.foundation@gmail.com
