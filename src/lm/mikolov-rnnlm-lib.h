///////////////////////////////////////////////////////////////////////
// 
//  Recurrent neural network based statistical language modeling toolkit
//  Version 0.3e
//  (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//           2015 Guoguo Chen
//           2015 Hainan Xu
// 
///////////////////////////////////////////////////////////////////////

#ifndef KALDI_LM_MIKOLOV_RNNLM_H_
#define KALDI_LM_MIKOLOV_RNNLM_H_

#define MAX_STRING 100
#define MAX_FILENAME_STRING 300

#include <string>
#include <vector>
#include "util/stl-utils.h"

typedef double real;		//  doubles for NN weights
typedef double direct_t;	//  doubles for ME weights;
//  TODO: check why floats are not enough for RNNME (convergence problems)

struct neuron {
  real ac;		// actual value stored in neuron
  real er;		// error value in neuron, used by learning algorithm
};

struct synapse {
  real weight;	// weight of synapse
};

struct vocab_word {
  int cn;
  char word[MAX_STRING];

  real prob;
  int class_index;
};

const unsigned int PRIMES[] = {108641969, 116049371, 125925907, 133333309,
  145678979, 175308587, 197530793, 234567803, 251851741, 264197411,
  330864029, 399999781,
  407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243,
  656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
  782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127,
  953085797, 985184539, 990122807};
const unsigned int PRIMES_SIZE  =  sizeof(PRIMES) / sizeof(PRIMES[0]);

const int MAX_NGRAM_ORDER = 20;

enum FileTypeEnum {TEXT, BINARY, COMPRESSED}; //  COMPRESSED not yet implemented

class CRnnLM{
 protected:
  char train_file[MAX_FILENAME_STRING];
  char valid_file[MAX_FILENAME_STRING];
  char test_file[MAX_FILENAME_STRING];
  char rnnlm_file[MAX_FILENAME_STRING];
  char lmprob_file[MAX_FILENAME_STRING];

  int rand_seed;
  int version;
  int filetype;

  int use_lmprob;
  real gradient_cutoff;

  real dynamic;

  real alpha;
  real starting_alpha;
  int alpha_divide;
  double logp, llogp;
  float min_improvement;
  int iter;
  int vocab_max_size;
  int vocab_size;
  int train_words;
  int train_cur_pos;
  int counter;

  int anti_k;

  real beta;

  int class_size;
  int **class_words;
  int *class_cn;
  int *class_max_cn;
  int old_classes;

  struct vocab_word *vocab;
  void sortVocab();
  int *vocab_hash;
  int vocab_hash_size;

  int layer0_size;
  int layer1_size;
  int layerc_size;
  int layer2_size;

  long long direct_size;
  int direct_order;
  int history[MAX_NGRAM_ORDER];

  int bptt;
  int bptt_block;
  int *bptt_history;
  neuron *bptt_hidden;
  struct synapse *bptt_syn0;

  int gen;

  int independent;

  struct neuron *neu0;		// neurons in input layer
  struct neuron *neu1;		// neurons in hidden layer
  struct neuron *neuc;		// neurons in hidden layer
  struct neuron *neu2;		// neurons in output layer

  struct synapse *syn0;		// weights between input and hidden layer
  struct synapse *syn1;		// weights between hidden and output layer
                            // (or hidden and compression if compression>0)
  struct synapse *sync;		// weights between hidden and compression layer
  direct_t *syn_d;		// direct parameters between input and output layer
                        // (similar to Maximum Entropy model parameters)

  // backup used in training:
  struct neuron *neu0b;
  struct neuron *neu1b;
  struct neuron *neucb;
  struct neuron *neu2b;

  struct synapse *syn0b;
  struct synapse *syn1b;
  struct synapse *syncb;
  direct_t *syn_db;

  // backup used in n-bset rescoring:
  struct neuron *neu1b2;

  unordered_map<std::string, float> unk_penalty;
  std::string unk_sym;

 public:

  int alpha_set, train_file_set;

  CRnnLM()		// constructor initializes variables
  {
    version = 10;
    filetype = TEXT;

    use_lmprob = 0;
    gradient_cutoff = 15;
    dynamic = 0;

    train_file[0] = 0;
    valid_file[0] = 0;
    test_file[0] = 0;
    rnnlm_file[0] = 0;

    alpha_set = 0;
    train_file_set = 0;

    alpha = 0.1;
    beta = 0.0000001;
    // beta = 0.00000;
    alpha_divide = 0;
    logp = 0;
    llogp = -100000000;
    iter = 0;

    min_improvement = 1.003;

    train_words = 0;
    vocab_max_size = 100;
    vocab_size = 0;
    vocab = (struct vocab_word *)calloc(vocab_max_size,
                                        sizeof(struct vocab_word));

    layer1_size = 30;

    direct_size = 0;
    direct_order = 0;

    bptt = 0;
    bptt_block = 10;
    bptt_history = NULL;
    bptt_hidden = NULL;
    bptt_syn0 = NULL;

    gen = 0;

    independent = 0;

    neu0 = NULL;
    neu1 = NULL;
    neuc = NULL;
    neu2 = NULL;

    syn0 = NULL;
    syn1 = NULL;
    sync = NULL;
    syn_d = NULL;
    syn_db = NULL;
    // backup
    neu0b = NULL;
    neu1b = NULL;
    neucb = NULL;
    neu2b = NULL;

    neu1b2 = NULL;

    syn0b = NULL;
    syn1b = NULL;
    syncb = NULL;
    // 

    rand_seed = 1;

    class_size = 100;
    old_classes = 0;

    srand(rand_seed);

    vocab_hash_size = 100000000;
    vocab_hash  =  (int *)calloc(vocab_hash_size, sizeof(int));
  }

  ~CRnnLM()		// destructor, deallocates memory
  {
    int i;

    if (neu0 != NULL) {
      free(neu0);
      free(neu1);
      if (neuc != NULL) free(neuc);
      free(neu2);

      free(syn0);
      free(syn1);
      if (sync != NULL) free(sync);

      if (syn_d != NULL) free(syn_d);

      if (syn_db != NULL) free(syn_db);

      // 
      free(neu0b);
      free(neu1b);
      if (neucb != NULL) free(neucb);
      free(neu2b);

      free(neu1b2);

      free(syn0b);
      free(syn1b);
      if (syncb != NULL) free(syncb);
      // 

      for (i = 0; i<class_size; i++) free(class_words[i]);
      free(class_max_cn);
      free(class_cn);
      free(class_words);

      free(vocab);
      free(vocab_hash);

      if (bptt_history != NULL) free(bptt_history);
      if (bptt_hidden != NULL) free(bptt_hidden);
      if (bptt_syn0 != NULL) free(bptt_syn0);

      // todo: free bptt variables too
    }
  }

  real random(real min, real max);

  void setRnnLMFile(const std::string &str);
  int getHiddenLayerSize() const { return layer1_size; }
  void setRandSeed(int newSeed) {rand_seed = newSeed; srand(rand_seed);}

  int getWordHash(const char *word);
  void readWord(char *word, FILE *fin);
  int searchVocab(const char *word);

  void saveWeights();			// saves current weights and unit activations
  void initNet();
  void goToDelimiter(int delim, FILE *fi);
  void restoreNet();
  void netReset();    // will erase just hidden layer state + bptt history
                      // + maxent history
                      // (called at end of sentences in the independent mode)

  void computeNet(int last_word, int word);
  void copyHiddenLayerToInput();

  void matrixXvector(struct neuron *dest, struct neuron *srcvec,
                     struct synapse *srcmatrix, int matrix_width,
                     int from, int to, int from2, int to2, int type);

  void restoreContextFromVector(const std::vector<float> &context_in);
  void saveContextToVector(std::vector<float> *context_out);

  float computeConditionalLogprob(
      std::string current_word,
      const std::vector<std::string> &history_words,
      const std::vector<float> &context_in,
      std::vector<float> *context_out);

  void setUnkSym(const std::string &unk);
  void setUnkPenalty(const std::string &filename);
  float getUnkPenalty(const std::string &word);
  bool isUnk(const std::string &word);
};

#endif
