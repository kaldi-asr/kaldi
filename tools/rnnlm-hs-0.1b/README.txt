1) OVERVIEW

This is a tool to estimate recurrent neural network language models on large amounts of text data. 

The source code is heavily based on the word2vec toolkit. Basically this is just a small extension of word2vec that makes possible to apply it for language modeling.
https://code.google.com/p/word2vec/

The main ideas, interface and maxent extension implementation come from RNNLM written by Tomas Mikolov:
http://rnnlm.org

The differences from the published Mikolov's RNNLM are hierarchical softmax and hogwild multithreading (both tricks are taken directly from word2vec). This makes possible to train RNNLM-HS (hierarchical softmax) on large corpora, e.g. billions of words. However, on small and average-sized corpora Tomas Mikolov's RNNLM works considerably better both in terms of entropy and WER. RNNLM-HS is also much faster in test time, which is useful for online ASR.

Please send you ideas and proposals regarding this tool to ilia@yandex-team.com (Ilya Edrenkin, Yandex LLC). Bugreports and fixes are also of course welcome.

2) USAGE EXAMPLES

A typical example to obtain a reasonable model on a large (~4 billion words) corpus in a couple of days on a 16-core machine:
./rnnlm -train corpus.shuf.split-train -valid corpus.shuf.split-valid -size 100 -model corpus.shuf.split-train.h100me5-1000.t16 -threads 16 -alpha 0.1 -bptt 4 -bptt-block 10 -maxent-order 5 -maxent-size 1000

Fine-tuning of an existing model on a smaller in-domain corpora:
./rnnlm -train corpus.indomain.split-train -valid corpus.indomain.split-valid -model corpus.shuf.split-train.h100me5-1000.t16 -threads 1 -bptt 0 -alpha 0.01 -recompute-counts 1

Obtaining individual logprobs for a set of test sentences:
./rnnlm -model corpus.shuf.split-train.h100me5-1000.t16 -test corpus.test

Interactive sampling from an existing model:
./rnnlm -model corpus.shuf.split-train.h100me5-1000.t16 -gen -10

3) USAGE ADVICE

- you don't need to repeat structural parameters (size, maxent-order, maxent-size) when using an existing model. They will be ignored. The vocab saved in the model will be reused.
- the vocabulary is built based on the training file on the first run of the tool for a particular model. The program will ignore sentences with OOVs in train time (or report them in test time).
- set the number of threads to the number of physical CPUs or less.
- vocabulary size plays very small role in the performance (it is logarithmic in the size of vocabulary due to the Huffman tree decomposition). Hidden layer size and the amount of training data are the main factors.
- the model will be written to file after a training epoch if and only if its validation entropy improved compared to the previous epoch.
- using multithreading together with unlimited BPTT (setting -bptt 0) may cause the net to diverge.
- unlike Mikolov's RNNLM which has -independent switch, this tool always considers sentences as independent and doesn't track global context.
- it is a good idea to shuffle sentences in the set before splitting them into training and validation sets (GNU shuf & split are one of the possible choices to do it). 

4) PARAMETERS

	-train <file>
		Use text data from <file> to train the model
	-valid <file>
		Use text data from <file> to perform validation and control learning rate
	-test <file>
		Use text data from <file> to compute logprobs with an existing model

Train, valid and test corpora. All distinct words that are found in the training file will be used for the nnet vocab, their counts will determine Huffman tree structure and remain fixed for this nnet. 
If you prefer using limited vocabulary (say, top 1 million words) you should map all other words to <unk> or another token of your choice. Limited vocabulary is usually a good idea if it helps you to have enough training examples for each word.

	-rnnlm <file>
		Use <file> to save the resulting language model

Will create <file> and <file>.nnet files (for storing vocab/counts in the text form and the net itself in binary form).
If the <file> and <file>.nnet already exist, the tool will attempt to load them instead of starting new training.

	-hidden <int>
		Set size of hidden layer; default is 100

Large (300 and more) hidden layers are slow; sometimes they even fail to learn well in combination with multithreading and large or unlimited BPTT depth.

	-bptt <int>
		Set length of BPTT unfolding; default is 3; set to 0 to disable truncation
	-bptt-block <int>
		Set period of BPTT unfolding; default is 10; BPTT is performed each bptt+bptt_block steps

Default parameters for BPTT are reasonable in the most cases. 

	-gen <int>
		Sampling mode; number of sentences to sample, default is 0 (off); enter negative number for interactive mode

Can complete a given prefix or generate a whole sentence (which corresponds to empty given prefix). Can be fun with a well trained model. Probably could be used for generating text for ngram variational approximation of the RNNLM, never tried it myself though.

	-threads <int>
		Use <int> threads (default 1)

The performance does not scale linearly with the number of threads (it is sublinear due to cache misses, false hogwild assumptions, etc).
Testing, validation and sampling are always performed by a single thread regardless of this setting.

	-min-count <int>
		This will discard words that appear less than <int> times; default is 0

Inherited from word2vec; not tested thoroughly. It is better to map rare words to <unk> by hand before training.

	-alpha <float>
		Set the starting learning rate; default is 0.1
	-maxent-alpha <float>
		Set the starting learning rate for maxent; default is 0.1

Maxent is somewhat prone to overfitting. Sometimes it makes sense to make maxent-alpha less than the 'main' alpha.

	-reject-threshold <float>
		Reject nnet and reload nnet from previous epoch if the relative entropy improvement on the validation set is below this threshold (default 0.997)
	-stop <float>
		Stop training when the relative entropy improvement on the validation set is below this threshold (default 1.003); see also -retry
	-retry <int>
		Stop training iff N retries with halving learning rate have failed (default 2)

The training schedule is as follows: if the validation set entropy improvement is less than <stop>, learning rate (both nnet and maxent) will start halving with each iteration.
In this mode, if the validation set entropy improvement is again less than <stop>, the retry counter will increment; when it reaches <retry> the training stops.
In addition to that, if the validation set entropy improvement is less than <reject-threshold> the nnet will be rejected and reloaded from the previous epoch.

	-debug <int>
		Set the debug mode (default = 2 = more info during training)

Inherited from word2vec. Set debug to 0 if you don't want to see speed statistics. 

	-direct-size <int>
		Set the size of hash for maxent parameters, in millions (default 0 = maxent off)
	-direct-order <int>
		Set the order of n-gram features to be used in maxent (default 3)

Maxent extension. Off by default. Speeds up convergence a lot, also improves entropy; the only drawback is memory demand, e.g. setting -direct-size 1000 will cost you ~4 GB for the nnet file.

	-beta1 <float>
		L2 regularisation parameter for RNNLM weights (default 1e-6)
	-beta2 <float>
		L2 regularisation parameter for maxent weights (default 1e-6)

Maxent is somewhat prone to overfitting. Sometimes it makes sense to make beta2 larger than beta1.

	-recompute-counts <int>
		Recompute train words counts, useful for fine-tuning (default = 0 = use counts stored in the vocab file)

Vocabulary counts are stored in the nnet file and used to reconstruct Huffman tree. When fine-tuning an existing model on a new corpus, use this option. New counts will not be saved, they are used for the fine-tuning session only.


5) FUTURE PLANS

Large amount of tricks that help training RNNs has been discussed in the literature and could be applied to his tool, but currently are not implemented in this release:

- Better initialization (spectral radius trick, Bayesian model selection)
- Using ReLU/softplus instead of sigmoid
- Momentum/NAG
- Second order methods, HF
- LSTM
- Adapting code for GPUs
- More efficient SGD parallelisation 

If you are interested in contribution feel free to participate.
