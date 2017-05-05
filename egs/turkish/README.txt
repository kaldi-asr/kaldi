### 
Turkish lexicon and phoneset by Duygu Altinok
###


### OVERVIEW
The package contains Turkish  lexicon and phoneset. Users are to provide the speech corpora. Here's what needed:
a. train - speech data and transription for training automatic speech recognition system (Kaldi ASR format)
b. test - speech data and transription for testing automatic speech recognition system (Kaldi ASR format)

A text corpus and language model in the directory asr_turkish/LM, and lexicon in the directory asr_turkish/lang



### TURKISH SPEECH CORPUS
Directory: asr_turkish/data/train
Files needed: text (training transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files). 
I used a small training corpus including voices of only 5 people including myself.

Directory: asr_turkish/data/test
Files needed: text (test transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files)
I used a small test corpus including voices of only  myself.



### TURKISH LANGUAGE MODEL
Directory: asr_turkish/LM
Files: small_text.txt, turkish.arpa

# /small_text.txt
Very small corpus.

# /turkish.arpa
A language model created from the small_text.txt. Very small language model for Turkish, supporting only 10 simple sentences.



### TURKISH LEXICON/PRONUNCIATION DICTIONARY
Directory: asr_turkish/lang
Files: lexicon.txt (lexicon), nonsilence_phones.txt (speech phones), optional_silence.txt (silence phone)
Description: lexicon contains words and their respective pronunciation, non-speech sound and noise in Kaldi format. 

