The FAME! Speech Corpus

The components of the Frisian data collection are speech and language resources gathered for building a large vocabulary ASR system for the Frisian language. Firstly, a new broadcast database is created by collecting recordings from the archives of the regional broadcaster Omrop Fryslân, and annotating them with various information such as the language switches and speaker details. The second component of this collection is a language model created on a text corpus with diverse vocabulary. Thirdly, a Frisian phonetic dictionary with the mappings between the Frisian words and phones is built to make the ASR viable for this under-resourced language. Finally, an ASR recipe is provided which uses all previous resources to perform recognition and present the recognition performances.

The Corpus consists of short utterances extracted from 203 audio segments of approximately 5 minutes long which are parts of various radio programs covering a time span of almost 50 years (1966-2015), adding a longitudinal dimension to the database. The content of the recordings are very diverse including radio programs about culture, history, literature, sports, nature, agriculture, politics, society and languages. The total duration of the manually annotated radio broadcasts sums up to 18 hours, 33 minutes and 57 seconds. The stereo audio data has a sampling frequency of 48 kHz and 16-bit resolution per sample. The available meta-information helped the annotators to identify these speakers and mark them either using their names or the same label (if the name is not known). There are 309 identified speakers in the FAME! Speech Corpus, 21 of whom appear at least 3 times in the database. These speakers are mostly program presenters and celebrities appearing multiple times in different recordings over years. There are 233 unidentified speakers due to lack of meta-information. The total number of word- and sentence-level code-switching cases in the FAME! Speech Corpus is equal to 3837. Music portions have been removed, except where these overlap with speech.

A full description of the FAME! Speech Corpus is provided in:

E. Yılmaz, H. van den Heuvel, J. Dijkstra, H. Van de Velde, F. Kampstra, J. Algra and D. van Leeuwen, "Open Source Speech and Language Resources for Frisian,"  In Proc. INTERSPEECH, pp. 1536-1540, San Francisco, CA, USA, Sept. 2016. 

Speaker clustering and verification corpus details are provided in:

E. Yılmaz, J. Dijkstra, H. Van de Velde, F. Kampstra, J. Algra, H. van den Heuvel and D. van Leeuwen, "Longitudinal Speaker Clustering and Verification Corpus with Code-switching Frisian-Dutch Speech," in Proc. INTERSPEECH, pp. 37-41 Stockholm, Sweden, August 2017.

Please check http://www.ru.nl/clst/datasets/ to get the FAME! Speech Corpus. The ASR scripts are in ./s5. The GMM-UBM and DNN-UBM SV scripts are in ./v1 and ./v2 respectively. 
