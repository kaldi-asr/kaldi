// author: Nikolay Malkovskiy (malkovskynv@gmail.com) 2019

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include "base/kaldi-common.h"
#include "nnet3/nnet-utils.h"
#include "rnnlm/rnnlm-compute-state.h"
#include "rnnlm/rnnlm-core-compute.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "rnnlm/rnnlm-training.h"
#include "util/common-utils.h"

#include <cmath>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program takes input of a trained RNNLM and\n"
        "generates text word by word by sampling the next it randomly based "
        "on\n"
        "RNNLM posterior distribution\n"
        "Usage:\n"
        " generate-text [options] <rnnlm> <word-embedding-matrix> <lang>\n"
        "e.g.:\n"
        " rnnlm-generate-text rnnlm/final.raw rnnlm/final.word_embedding "
        "data/lang > data/rnnlm_generated/text.txt\n";
    std::string use_gpu = "no";
    bool batchnorm_test_mode = true, dropout_test_mode = true;
    std::string num_words_str;
    std::string srand_offset_str;

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
    po.Register("num-words", &num_words_str,
                "Total number of words to generate (excluding begin of "
                "sentence and end of sentence)");
    po.Register("srand-offset", &srand_offset_str,
                "Adds this number to the time(NULL) when initializing srand");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (opts.bos_index == -1 || opts.eos_index == -1) {
      KALDI_ERR << "You must set --bos-symbol and --eos-symbol options";
    }

    std::istringstream iss(num_words_str);
    int64 num_words = -1;
    iss >> num_words;
    if (num_words < 0 || iss.fail()) {
      KALDI_ERR << "You must set --num-words as a positive integer";
    }
    iss = std::istringstream(srand_offset_str);
    int64 srand_offset = 0;
    iss >> srand_offset;
    srand(time(NULL) + srand_offset);

    std::string rnnlm_rxfilename = po.GetArg(1),
                word_embedding_rxfilename = po.GetArg(2),
                lang_dir = po.GetArg(3);

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    KALDI_ASSERT(IsSimpleNnet(rnnlm));
    if (batchnorm_test_mode) SetBatchnormTestMode(true, &rnnlm);
    if (dropout_test_mode) SetDropoutTestMode(true, &rnnlm);

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    std::vector<std::string> words;
    std::ifstream words_txt(lang_dir + "/words.txt");
    if (words_txt.is_open()) {
      std::string w;
      int w_id;
      while (!words_txt.eof()) {
        words_txt >> w >> w_id;
        KALDI_ASSERT(w_id == words.size());
        // if (w_id != words.size() ) {
        //  std::cerr << w_id << " " << words.size() << " " << w << std::endl;
        //  exit(1);
        //}
        words.push_back(w);
      }
    } else {
      KALDI_ERR << "words.txt is not found in lang.";
      exit(1);
    }

    float *Probs = new float[words.size() - 1];

    std::cerr << "Done reading models." << std::endl;
    std::cerr << "Starting text generation." << std::endl;

    while (num_words > 0) {
      RnnlmComputeState rnnlm_compute_state(info, opts.bos_index);
      // 0 is for eps
      int cur_word = 0;
      while (cur_word != opts.eos_index && num_words--) {
        // Computing softmax
        // Can probably use GetLogProbOfwords but i don't know kaldi very good
        // and GetLogProbOfWords is not used anywhere in kaldi
        float s = 0;
        int word_id;
        for (word_id = 1; word_id < words.size(); ++word_id) {
          Probs[word_id - 1] = exp(rnnlm_compute_state.LogProbOfWord(word_id));
          s += Probs[word_id - 1];
        }
        for (word_id = 1; word_id < words.size(); ++word_id) {
          Probs[word_id - 1] /= s;
        }
        // generating uniform random real in [0, 1]. Probably not the best way
        // to do it.
        double r = (double)(std::rand()) / RAND_MAX;
        // Choosing the next word accoring to current posterior distribution and
        // generated random
        word_id = 0;
        s = 0;
        while (s <= r && word_id < words.size() - 1) {
          s += Probs[word_id++];
        }
        cur_word = word_id;
        if (cur_word == opts.bos_index) {
          continue;
        } else if (word_id == opts.eos_index) {
          std::cout << std::endl;
          num_words++;
        } else {
          std::cout << words[cur_word] << " ";
          rnnlm_compute_state.AddWord(cur_word);
        }
      }
    }
#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
