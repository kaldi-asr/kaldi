//  word2vec: Copyright 2013 Google Inc. All Rights Reserved.
//  RNNLM extension: Copyright 2014 Yandex LLC. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>

#define MAX_STRING 1024
#define MAX_SENTENCE_LENGTH 10000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

// #define DOUBLE
// #define DEBUG

#ifdef DOUBLE        // Precision of float numbers
typedef double real;
#else
typedef float real;
#endif

#define MAX_GRAD 15.0f
#define MIN_GRAD -MAX_GRAD

#define MAX_NGRAM_ORDER 10

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct tree_node {
  int child0;
  int child1;
};
struct tree_node *tree;

struct {
  real *syn0, *syn1, *synRec, *synMaxent;
} nnet = { NULL, NULL, NULL, NULL };

char train_file[MAX_STRING], valid_file[MAX_STRING], model_file[MAX_STRING], model_file_nnet[MAX_STRING+5], test_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, min_count = 0, min_reduce = 0, num_threads = 1, recompute_train_counts = 0;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, valid_words = 0, word_count_actual = 0, file_size = 0, valid_file_size = 0;
real alpha = 0.1;
real maxent_alpha = 0.1;
real *expTable;
clock_t start;
real sumlogprob = 0, sumlogprob_valid = 0;
real stop = 1.003;
real reject_threshold = 0.997;
int max_retry = 2;
int bptt = 3;
int bptt_block = 10;
unsigned long long counter = 0;
int maxent_order = 3;
long long maxent_hash_size = 0;
real beta = 1e-6;
real maxent_beta = 1e-6;
int gen = 0;

const unsigned long long int PRIMES[]={108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned long long int PRIMES_SIZE=sizeof(PRIMES)/sizeof(PRIMES[0]);

inline static void ApplySigmoid(real* neu, long long layer_size) { 
  for (int c = 0; c < layer_size; c++) {
    neu[c] = exp(neu[c])/(1 + exp(neu[c]));
  }
}

inline static void MultiplySigmoidDerivative(real* neu, long long layer_size, real* neu_e) { 
  for (int c = 0; c < layer_size; ++c) {
    neu_e[c] *= neu[c] * (1 - neu[c]);
    neu_e[c] = neu_e[c] < MAX_GRAD ? neu_e[c] : MAX_GRAD;
    neu_e[c] = neu_e[c] > MIN_GRAD ? neu_e[c] : MIN_GRAD;
  }
}


// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size *= 1.5; // was += 1000, modified to have fewer reallocations
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short unique binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  tree = calloc(vocab_size, sizeof(struct tree_node));

  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0 && count[pos1] < count[pos2]) {
      min1i = pos1;
      pos1--;
    } else {
      min1i = pos2;
      pos2++;
    }

    if (pos1 >= 0 && count[pos1] < count[pos2]) {
      min2i = pos1;
      pos1--;
    } else {
      min2i = pos2;
      pos2++;
    }

    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    tree[a].child0 = min1i - vocab_size;
    tree[a].child1 = min2i - vocab_size;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      fprintf(stderr, "%lldK%c", train_words / 1000, 13);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    fprintf(stderr, "Vocab size: %lld\n", vocab_size);
    fprintf(stderr, "Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(model_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(model_file, "rb");
  if (fin == NULL) {
    fprintf(stderr, "Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();

  if(recompute_train_counts) { // If training file changed, e.g. in fine-tuning
    FILE *fi = fopen(train_file, "rb");
    if (fi == NULL) {
      fprintf(stderr, "ERROR: training data file not found!\n");
      exit(1);
    }
    train_words = 0;
    while (1) {
      ReadWordIndex(fi);
      ++train_words;
      if (feof(fi)) break;
    }    
    fclose(fi);
  }

  if (debug_mode > 0) {
    fprintf(stderr, "Vocab size: %lld\n", vocab_size);
    fprintf(stderr, "Words in train file: %lld\n", train_words);
  }
  if(test_file[0] != 0 || gen != 0) return;

  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    fprintf(stderr, "ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&nnet.syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (nnet.syn0 == NULL) {fprintf(stderr, "Memory allocation failed (syn0)\n"); exit(1);}

  a = posix_memalign((void **)&nnet.syn1, 128, (long long)layer1_size * vocab_size * sizeof(real));
  if (nnet.syn1 == NULL) {fprintf(stderr, "Memory allocation failed (syn1)\n"); exit(1);}

  a = posix_memalign((void **)&nnet.synRec, 128, (long long)layer1_size * layer1_size * sizeof(real));
  if (nnet.synRec == NULL) {fprintf(stderr, "Memory allocation failed (synRec)\n"); exit(1);}

  if(maxent_hash_size != 0) {
    a = posix_memalign((void **)&nnet.synMaxent, 128, (long long)maxent_hash_size * sizeof(real));
    if (nnet.synMaxent == NULL) {fprintf(stderr, "Memory allocation failed (synMaxent)\n"); exit(1);}
    memset(nnet.synMaxent, 0, (long long) maxent_hash_size * sizeof(real));
  } else {
    maxent_order = 0;
  }

  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) nnet.syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) nnet.syn1[a * layer1_size + b] = 0;
  for (a = 0; a < layer1_size; a++) for (b = 0; b < layer1_size; b++) nnet.synRec[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long d, word = -1, last_word, sentence_length = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l2;
  real f, g;
  clock_t now;
  real *neu1, *neu1e;
  int full_block = bptt_block + bptt;
  posix_memalign((void **)&neu1, 128, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real));
  posix_memalign((void **)&neu1e, 128, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real));

  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
#ifdef DEBUG
  real my_sumlogprob = 0;
#endif
  if((long long)id != 0) while(word != 0 && !feof(fi)) { // skipping to the next newline
      word = ReadWordIndex(fi); 
    }

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if (debug_mode > 1) {
        now=clock();
        fprintf(stderr, "%cAlpha: %f  ME-alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk\t", 13, alpha, maxent_alpha,
		word_count_actual / (real)(train_words + 1) * 100,
		word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
    }
    if (feof(fi) || word_count > train_words / num_threads) break;
   
    sen[0] = 0; // <s> token -- beginning of sentence
    int good = 1;
    sentence_length = 1;
    while(sentence_length < MAX_SENTENCE_LENGTH) {
      word = ReadWordIndex(fi);
      ++word_count;
      sen[sentence_length] = word;
      if (feof(fi) || word == 0) break;
      if (word == -1) good = 0;
      ++sentence_length;
    }

    if(good == 0) continue;
    if(sentence_length == 1 && feof(fi)) break;
    
    memset(neu1e, 0, (long long)layer1_size * sentence_length * sizeof(real)); // clear gradients  
    memset(neu1, 0, (long long)layer1_size * sentence_length * sizeof(real)); // clear activations
#ifdef DEBUG
    real sentence_logprob = 0.0;
#endif

    for(int input = 0; input < sentence_length; ++input) {
      // Forward pass (not including final softmax)  
      if (input != 0) { 
	for(int c = 0; c < layer1_size; ++c) {
	  for(int d = 0; d < layer1_size; ++d) { 
	    neu1[input*layer1_size + c] += nnet.synRec[c*layer1_size + d] * neu1[(input-1)*layer1_size + d];  // Recurrent hidden->hidden activation
	  }
	}
      }
      last_word = sen[input];
      for(int c = 0; c < layer1_size; ++c) {
	neu1[input*layer1_size + c] += nnet.syn0[last_word*layer1_size + c]; // Input to hidden
      }
      ApplySigmoid(neu1+layer1_size*input, layer1_size);
    }
    
    for(int target = 1; target <= sentence_length; ++target) {
      // Forward pass (softmax)
      word = sen[target];
      long long feature_hashes[MAX_NGRAM_ORDER] = {0};
      if(maxent_order) {
	for(int order = 0; order < maxent_order && target - order >= 0; ++order) {
	  feature_hashes[order] = PRIMES[0]*PRIMES[1];    	    
	  for (int b = 1; b <= order; ++b) feature_hashes[order] += PRIMES[(order*PRIMES[b]+b) % PRIMES_SIZE]*(unsigned long long)(sen[target-b]+1);
	  feature_hashes[order] = feature_hashes[order] % (maxent_hash_size - vocab_size);
	}
      }
      for (d = 0; d < vocab[word].codelen; d++) {
	// Propagate hidden -> output
	f = 0.0;
	l2 = vocab[word].point[d] * layer1_size;
	for(int c = 0; c < layer1_size; ++c) {
	  f += neu1[layer1_size*(target - 1) + c] * nnet.syn1[l2 + c];
	}
	for(int order = 0; order < maxent_order && target - order >= 0; ++order) {
	  f += nnet.synMaxent[feature_hashes[order] + vocab[word].point[d]];
	}
#ifdef DEBUG
	sentence_logprob += log10(1+(vocab[word].code[d] == 1 ? exp(f) : exp(-f)));
#endif
	f = exp(f)/(1+exp(f)); // sigmoid
	g = (1 - vocab[word].code[d] - f); 
	g = g > MAX_GRAD ? MAX_GRAD : g;
	g = g < MIN_GRAD ? MIN_GRAD : g;
	real g_alpha = g * alpha; // 'g_alpha' is the gradient multiplied by the learning rate
	real g_maxentalpha = g * maxent_alpha;

	// Propagate errors output -> hidden
	for(int c = 0; c < layer1_size; ++c) {
	  neu1e[layer1_size * (target - 1) + c] += g_alpha * nnet.syn1[l2 + c];
        }

	// Learn weights hidden -> output
	for(int c = 0; c < layer1_size; ++c) {
	  nnet.syn1[l2 + c] += g_alpha * neu1[layer1_size*(target - 1) + c] - beta * nnet.syn1[l2 + c];
        }
	for(int order = 0; order < maxent_order && target - order >= 0; ++order) {
          nnet.synMaxent[feature_hashes[order] + vocab[word].point[d]] += g_maxentalpha - maxent_beta * nnet.synMaxent[feature_hashes[order] + vocab[word].point[d]];
        }
      }
    }
#ifdef DEBUG
    my_sumlogprob += sentence_logprob;
#endif
    // Backpropagation through time pass
    int my_bptt = 0;
    for(int input = sentence_length - 1; input >= 0; --input) {
	MultiplySigmoidDerivative(neu1+layer1_size*input, layer1_size, neu1e+layer1_size*input);  
	last_word = sen[input];

	for(int c = 0; c < layer1_size; ++c) {
	  nnet.syn0[layer1_size*last_word + c] += neu1e[layer1_size*input + c] - beta * nnet.syn0[layer1_size*last_word + c]; // Input weight update
	}

	long long word_num = word_count - (input - sentence_length);
	if(full_block == 0 || word_num % full_block == 0) {
	  my_bptt = bptt;
	}
	if(input > 0 && (bptt == 0 || my_bptt > 0 )) {
	  // Work with recurrent weights: backpropagate
	  for(int c = 0; c < layer1_size; ++c) {
	    for(int d = 0; d < layer1_size; ++d) {
	      neu1e[(input-1)*layer1_size + d] += nnet.synRec[c*layer1_size + d] * neu1e[input*layer1_size + c];  // Recurrent hidden->hidden backprop
	    }
	  }
	  --my_bptt;
	}
    } // End BPTT loop

    for(int input = sentence_length - 1; input > 0; --input) {
      // Work with recurrent weights: update
 	for(int c = 0; c < layer1_size; ++c) {
	  for(int d = 0; d < layer1_size; ++d) { 
	    nnet.synRec[c*layer1_size + d] += neu1e[input*layer1_size + c] * neu1[(input-1)*layer1_size + d] - beta * nnet.synRec[c*layer1_size + d]; // Recurrent hidden->hidden weight update
	  }
	}
     }

  } // End main training loop

#ifdef DEBUG
  if((long long)id == 0) fprintf(stderr, "Train Entropy (thread %lld, word count %lld) %f\t", (long long)id, word_count, my_sumlogprob/log10(2)/(real)word_count);
#endif

  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

real EvaluateModel(char* filename, int printLoglikes) {
  long long d, word = -1, last_word, sentence_length = 0;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  long long l2;
  real f;
  real *neu1;
  posix_memalign((void **)&neu1, 128, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real));
  memset(neu1, 0, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real));

  FILE *fi = fopen(filename, "rb");
  real my_sumlogprob = 0;

  while (1) {
    if (feof(fi)) break;
   
    sen[0] = 0;
    int good = 1;
    sentence_length = 1;
    while(sentence_length < MAX_SENTENCE_LENGTH) {
      word = ReadWordIndex(fi);
      sen[sentence_length] = word;
      if (feof(fi) || word == 0) break;
      if( word == -1) good = 0;
      ++sentence_length;
    }
    if(good == 0) {
      if(printLoglikes) printf("OOV\n");
      continue;
    }
    if(sentence_length == 1 && feof(fi)) break;
    real sentence_logprob = 0.0;

    memset(neu1, 0, (long long)layer1_size * sentence_length * sizeof(real));

    for(int input = 0; input < sentence_length; ++input) {
      // Forward pass (not including final softmax)  
      if (input != 0) { 
	for(int c = 0; c < layer1_size; ++c) {
	  for(int d = 0; d < layer1_size; ++d) { neu1[input*layer1_size + c] += nnet.synRec[c*layer1_size + d] * neu1[(input-1)*layer1_size + d]; } // Recurrent hidden->hidden activation
	}
      }
      last_word = sen[input];
      for(int c = 0; c < layer1_size; ++c) {
	neu1[input*layer1_size + c] += nnet.syn0[last_word*layer1_size + c]; // Input to hidden
      }
      ApplySigmoid(neu1+layer1_size*input, layer1_size);
    }

    for(int target = 1; target <= sentence_length; ++target) {
      // Forward pass (softmax)
      word = sen[target];
      long long feature_hashes[MAX_NGRAM_ORDER] = {0};
      if(maxent_order) {
	for(int order = 0; order < maxent_order && target - order >= 0; ++order) {
	  feature_hashes[order] = PRIMES[0]*PRIMES[1];    	    
	  for (int b = 1; b <= order; ++b) feature_hashes[order] += PRIMES[(order*PRIMES[b]+b) % PRIMES_SIZE]*(unsigned long long)(sen[target-b]+1);
	  feature_hashes[order] = feature_hashes[order] % (maxent_hash_size - vocab_size);
	}
      }
      real logprob = 0.0;

      int maxent_present = maxent_order;
      // Check if we should exclude some ME features that were probably not learned
      for(int order = 0; order < maxent_order && target - order >= 0; ++order) {
	for (d = 0; d < vocab[word].codelen; d++) {
	  if (nnet.synMaxent[feature_hashes[order] + vocab[word].point[d]] == 0) { 
	    // Make ME hash act a Bloom filter: if a weight is zero, it was probably never touched by training and this (an higher) ngrams should not be considered for this target.
	    maxent_present = order;
	    break;
	  }
	}
      }      

      for (d = 0; d < vocab[word].codelen; d++) {
	// Propagate hidden -> output
	f = 0.0;
	l2 = vocab[word].point[d] * layer1_size;
	for(int c = 0; c < layer1_size; ++c) {
	  f += neu1[layer1_size*(target - 1) + c] * nnet.syn1[l2 + c];
	}
	for(int order = 0; order < maxent_present && target - order >= 0; ++order) {
	  f += nnet.synMaxent[feature_hashes[order] + vocab[word].point[d]];
	}
	logprob += log10(1+(vocab[word].code[d] == 1 ? exp(f) : exp(-f)));	
      }
      sentence_logprob += logprob;
      ++counter;
    }
    if(printLoglikes) printf("%f\n", -sentence_logprob);
    my_sumlogprob += sentence_logprob;
  }
  fclose(fi);
  free(neu1);
  return my_sumlogprob;
}

void Sample(int num_sentences, int interactive) {

  long long last_word;
  long long sen[MAX_SENTENCE_LENGTH + 1];
  long long l2;
  real f;
  real *neu1;
  int begin = 0;
  posix_memalign((void **)&neu1, 128, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real));
  sen[0] = 0;
  if(interactive) {
    printf("Enter the phrase to be continued:\n");
    while(1) {
      int word = ReadWordIndex(stdin);
      if(word == 0) break;
      if(word == -1) word = SearchVocab("<unk>");
      ++begin;
      sen[begin] = word;      
    }

  }

  int sentence = 0;
  while (sentence < num_sentences) {
    memset(neu1, 0, (long long)layer1_size * MAX_SENTENCE_LENGTH * sizeof(real)); // clean activations

    for(int i = 1; i <= begin; ++i) printf("%s ", vocab[sen[i]].word);
    if(begin) printf("| ");
    int input = 0;
    real logprob = 0.0;
    while(1) {

      if (input != 0) { 
	for(int c = 0; c < layer1_size; ++c) {
	  for(int d = 0; d < layer1_size; ++d) { neu1[input*layer1_size + c] += nnet.synRec[c*layer1_size + d] * neu1[(input-1)*layer1_size + d]; } // Recurrent hidden->hidden activation
	}
      }
      last_word = sen[input];
      for(int c = 0; c < layer1_size; ++c) {
	neu1[input*layer1_size + c] += nnet.syn0[last_word*layer1_size + c]; // Input to hidden
      }
      ApplySigmoid(neu1+layer1_size*input, layer1_size);
    
      if(input < begin) {
	++input;
	continue;
      }

      long long feature_hashes[MAX_NGRAM_ORDER] = {0};

      if(maxent_order) {
	for(int order = 0; order < maxent_order && input + 1 >= order; ++order) {
	  feature_hashes[order] = PRIMES[0]*PRIMES[1];    	    
	  for (int b = 1; b <= order; ++b) feature_hashes[order] += PRIMES[(order*PRIMES[b]+b) % PRIMES_SIZE]*(unsigned long long)(sen[input+1-b]+1);
	  feature_hashes[order] = feature_hashes[order] % (maxent_hash_size - vocab_size);
	}
      }

      int node = vocab_size - 2;
      while(node > 0) {
	// Propagate hidden -> output
	f = 0.0;
	l2 = node * layer1_size;
	for(int c = 0; c < layer1_size; ++c) {
	  f += neu1[input*layer1_size + c] * nnet.syn1[l2 + c];
	}
	for(int order = 0; order < maxent_order && input + 1 >= order; ++order) {
	  f += nnet.synMaxent[feature_hashes[order] + node];
	}
	f = exp(f)/(1+exp(f)); // sigmoid
	real random = rand() / (real)RAND_MAX;
	if (f > random) {
	  node = tree[node].child0; 
	  logprob += log10(f);
	} else {
	  node = tree[node].child1; 
	  logprob += log10(1-f);
	}
      }
      ++input;
      sen[input] = node + vocab_size;
      printf("%s ", vocab[sen[input]].word);
      if(sen[input] == 0 || input == MAX_SENTENCE_LENGTH) {
	printf("%f %f\n", logprob, logprob /(input-begin));
	break;
      }
    }
    ++sentence;
  }
  free(neu1);
}

void SaveNnet() {
  SaveVocab();  
  FILE* fo = fopen(model_file_nnet, "wb");

  fwrite(&layer1_size, sizeof(long long), 1, fo);
  fwrite(&maxent_hash_size, sizeof(long long), 1, fo);
  fwrite(&maxent_order, sizeof(int), 1, fo);

  fwrite(nnet.syn0, sizeof(real), layer1_size*vocab_size, fo);
  fwrite(nnet.syn1, sizeof(real), layer1_size*vocab_size, fo);
  fwrite(nnet.synRec, sizeof(real), layer1_size*layer1_size, fo);
  if(maxent_hash_size) fwrite(nnet.synMaxent, sizeof(real), maxent_hash_size, fo);
  fclose(fo);
}

void LoadNnet() {
  ReadVocab();
  FILE *fo = fopen(model_file_nnet, "rb");
  
  fread(&layer1_size, sizeof(long long), 1, fo);
  fread(&maxent_hash_size, sizeof(long long), 1, fo);
  fread(&maxent_order, sizeof(int), 1, fo);

  InitNet();
  fread(nnet.syn0, sizeof(real), layer1_size*vocab_size, fo);
  fread(nnet.syn1, sizeof(real), layer1_size*vocab_size, fo);
  fread(nnet.synRec, sizeof(real), layer1_size*layer1_size, fo);
  if(maxent_hash_size) fread(nnet.synMaxent, sizeof(real), maxent_hash_size, fo);
  fclose(fo);
}

void FreeNnet() {
  free(nnet.syn0);
  free(nnet.syn1);
  free(nnet.synRec);
  if(maxent_hash_size != 0) free(nnet.synMaxent);
}


void TrainModel() {
  long a;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (model_file[0] == 0) return;
  int iter = 0;

  FILE *t1 = fopen(model_file, "rb");
  FILE *t2 = fopen(model_file_nnet, "rb");
  if(t1 != NULL && t2 != NULL) {
    fclose(t1);
    fclose(t2);
    fprintf(stderr, "Restoring nnet from existing files %s, %s\n", model_file, model_file_nnet);
    LoadNnet();
  } else {
    LearnVocabFromTrainFile();
    if(maxent_hash_size) {
      maxent_hash_size *= 1000000;
      maxent_hash_size -= maxent_hash_size % vocab_size;
    }
    InitNet();
    SaveNnet();
  } 

  if(test_file[0] != 0) {
    counter = 0;
    real sumlogprob = EvaluateModel(test_file, 1);
    fprintf(stderr, "Test entropy %f\n", sumlogprob/log10(2)/(real)counter);
    return;
  }

  if(gen > 0) {
    Sample(gen, 0);
    return;
  } else if(gen < 0) {
    while(1) {
      Sample(-gen, 1);
    }
    return;
  }

  fprintf(stderr, "Starting training using file %s\n", train_file);

  FILE *fi = fopen(valid_file, "rb");
  valid_words = 0;
  while (1) {
    ReadWordIndex(fi);
    ++valid_words;
    if (feof(fi)) break;
  }    
  valid_file_size = ftell(fi);
  fclose(fi);

  real old_entropy = 1e99;
  real entropy;
  real diff = 1e99;
  int retry = 0;
  int decay = 0;
  while(retry < max_retry) {
    if(iter  != 0) {
      if(decay) {
	alpha /= 2.0;
	maxent_alpha /= 2.0;
      }
      word_count_actual = 0;
      counter = 0;
      start = clock();
      for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
      for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
    fprintf(stderr, "Iteration %d\t", iter);
    sumlogprob_valid = 0;
    counter = 0;
    sumlogprob_valid = EvaluateModel(valid_file, 0);
    entropy = sumlogprob_valid/log10(2)/(real)counter;
    fprintf(stderr, "Valid Entropy %f", entropy);
    ++iter;

    diff = old_entropy/entropy;  
    if (isnan(entropy) || isinf(entropy) || diff < stop) {
      if (decay == 1) {
	++retry;
	fprintf(stderr, "\tRetry %d/%d", retry, max_retry);
      } else {
	decay = 1;
	fprintf(stderr, "\tDecay started");
      }
      if(isnan(entropy) || isinf(entropy) || diff < reject_threshold) {
	fprintf(stderr, "\tNnet rejected");
	FreeNnet();
	int debug_ = debug_mode;
	debug_mode = 0;
	LoadNnet();
	debug_mode = debug_;
      }
    }
    fprintf(stderr, "\n");

    if(diff > 1.0) {  
      SaveNnet();
      old_entropy = entropy;
    }
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

void CopyOrFail(char *output, const char *input) {
  if (strlen(input) >= MAX_STRING ) {
    fprintf(stderr, "The string %s is too long! Probably too deep directory"
        "structure. Either patch the rnnlm.c or change the directory path\n",
        input);
  } else {
    strcpy(output, input);
  }
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("RNNLM based on WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-valid <file>\n");
    printf("\t\tUse text data from <file> to perform validation and control learning rate\n");
    printf("\t-test <file>\n");
    printf("\t\tUse text data from <file> to compute logprobs with an existing model\n");
    printf("\t-rnnlm <file>\n");
    printf("\t\tUse <file> to save the resulting language model\n");
    printf("\t-hidden <int>\n");
    printf("\t\tSet size of hidden layer; default is 100\n");
    printf("\t-bptt <int>\n");
    printf("\t\tSet length of BPTT unfolding; default is 3; set to 0 to disable truncation\n");
    printf("\t-bptt-block <int>\n");
    printf("\t\tSet period of BPTT unfolding; default is 10; BPTT is performed each bptt+bptt_block steps\n");
    printf("\t-gen <int>\n");
    printf("\t\tSampling mode; number of sentences to sample, default is 0 (off); enter negative number for interactive mode\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 0\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.1\n");
    printf("\t-maxent-alpha <float>\n");
    printf("\t\tSet the starting learning rate for maxent; default is 0.1\n");
    printf("\t-reject-threshold <float>\n");
    printf("\t\tReject nnet and reload nnet from previous epoch if the relative entropy improvement on the validation set is below this threshold (default 0.997)\n");
    printf("\t-stop <float>\n");
    printf("\t\tStop training when the relative entropy improvement on the validation set is below this threshold (default 1.003); see also -retry\n");
    printf("\t-retry <int>\n");
    printf("\t\tStop training iff N retries with halving learning rate have failed (default 2)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-direct <int>\n");
    printf("\t\tSet the size of hash for maxent parameters, in millions (default 0 = maxent off)\n");
    printf("\t-direct-order <int>\n");
    printf("\t\tSet the order of n-gram features to be used in maxent (default 3)\n");
    printf("\t-beta1 <float>\n");
    printf("\t\tL2 regularisation parameter for RNNLM weights (default 1e-6)\n");
    printf("\t-beta2 <float>\n");
    printf("\t\tL2 regularisation parameter for maxent weights (default 1e-6)\n");
    printf("\t-recompute-counts <int>\n");
    printf("\t\tRecompute train words counts, useful for fine-tuning (default = 0 = use counts stored in the vocab file)\n");
    printf("\nExamples:\n");
    printf("./rnnlm -train data.txt -valid valid.txt -rnnlm result.rnnlm -debug 2 -hidden 200\n\n");
    return 0;
  }
  model_file[0] = 0;
  test_file[0] = 0;
  if ((i = ArgPos((char *)"-hidden", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) CopyOrFail(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-valid", argc, argv)) > 0) CopyOrFail(valid_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-test", argc, argv)) > 0) CopyOrFail(test_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-bptt", argc, argv)) > 0) bptt = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-bptt-block", argc, argv)) > 0) bptt_block = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-maxent-alpha", argc, argv)) > 0) maxent_alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-reject-threshold", argc, argv)) > 0) reject_threshold = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-stop", argc, argv)) > 0) stop = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-retry", argc, argv)) > 0) max_retry = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-rnnlm", argc, argv)) > 0) {
    CopyOrFail(model_file, argv[i + 1]);
    CopyOrFail(model_file_nnet, argv[i + 1]);
    strcat(model_file_nnet, ".nnet");
  }
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-direct", argc, argv)) > 0) maxent_hash_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-direct-order", argc, argv)) > 0) maxent_order = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-beta1", argc, argv)) > 0) beta = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-beta2", argc, argv)) > 0) maxent_beta = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-gen", argc, argv)) > 0) gen = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-recompute-counts", argc, argv)) > 0) recompute_train_counts = atoi(argv[i + 1]);


  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  TrainModel();
  return 0;
}
