// rnnlm/rnnlm-utils-test.cc

#include <math.h>
#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

void UnitTestSampleConvergence() {
  // randomly generate unigrams
  int n = rand() % 10000 + 1000;
  vector<BaseFloat> selection_probs;
  BaseFloat prob = 0;
  BaseFloat prob_sum = 0;
  for (int i = 0; i < n; i++) {
    prob = RandUniform();
    prob_sum += prob;
    selection_probs.push_back(prob);
  }
  // KALDI_LOG << "Before Normalize, probs are:";
  for (int i = 0; i < n; i++) {
    selection_probs[i] /= prob_sum;
    // KALDI_LOG << selection_probs[i];
  }
  // generate a outputs_set with ones and with size < k
  int ones_size = rand() % (n / 2) + 1;
  // KALDI_LOG << "Size of ones_set is " << ones_size;
  std::set<int> outputs_set;
  for (int i = 0; i < ones_size; i++) {
    outputs_set.insert(rand() % n);
  }
  // generate a random number k from 1 to n
  int k = 2 * ones_size;
  // KALDI_LOG << "select " << k << " items from " << n << " unigrams";
  NormalizeVec(k, outputs_set, &selection_probs);
  /* 
  // check that the sum of selection_probs equal to k (passed)
  BaseFloat sum1 = 0;
  for (int i = 0; i < selection_probs.size(); i++) {
    sum1 += selection_probs[i];
    KALDI_LOG << sum1;
  }
  KALDI_ASSERT(ApproxEqual(sum1, k));
  */
  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }
  // normalize the selection_probs
  BaseFloat sum = 0;
  for (int i = 0; i < u.size(); i++) {
    sum += selection_probs[i];
  }
  for (int i = 0; i < u.size(); i++) {
    selection_probs[i] /= sum;
  }

  vector<BaseFloat> samples_probs(u.size(), 0);
  int count = 0;
  for (int i = 0; ; i++) {
    count++;
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
    for (int j = 0; j < samples.size(); j++) {
      samples_probs[samples[j]] += 1;
    }
    for (int j = 0; j < samples_probs.size(); j++) {
      samples_probs[j] /= (count * k);
    }
    // update Euclidean distance between the two pdfs
    BaseFloat distance = 0;
    for (int j = 0; j < u.size(); j++) {
      distance += pow(samples_probs[j] - selection_probs[j], 2);
    }
    distance = sqrt(distance);
    // if the Euclidean distance is small enough, break the loop
    if (distance < 0.05) {
      KALDI_LOG << "test of the sampling convergence is passed.";
      break;
    }
  }
}

// test that probabilities 1.0 are always sampled
void UnitTestSampleWithProbOne(int iters) {
  // generate unigrams
  int n = rand() % 200 + 5;
  vector<BaseFloat> selection_probs;
  BaseFloat prob = 0;
  BaseFloat prob_sum = 0;
  for (int i = 0; i < n; i++) {
    prob = RandUniform();
    prob_sum += prob;
    selection_probs.push_back(prob);
  }
  for (int i = 0; i < n; i++) {
    selection_probs[i] /= prob_sum;
  }

  // generate a outputs_set with ones and with size < k
  int ones_size = rand() % (n / 2) + 1;
  // std::cout << "size of ones_set is " << ones_size << std::endl;
  std::set<int> outputs_set;
  for (int i = 0; i < ones_size; i++) {
    outputs_set.insert(rand() % n);
  }

  // generate a random number k from 1 to n
  int k = 2 * ones_size;
  // std::cout << "select " << k << " items" << std::endl;
  NormalizeVec(k, outputs_set, &selection_probs);

  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }

  int N = iters;
  for (int i = 0; i < N; i++) {
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
    // assert every item in outputs_set is sampled
    for (set<int>::iterator it = outputs_set.begin(); it != outputs_set.end(); ++it) {
      KALDI_ASSERT(std::find(samples.begin(), samples.end(), *it) != \
          samples.end());
    }
  }
}

void UnitTestSamplingTime(int iters) {
  // generate unigrams
  int n = rand() % 200 + 5;
  vector<BaseFloat> selection_probs;
  BaseFloat prob = 0;
  BaseFloat prob_sum = 0;
  for (int i = 0; i < n; i++) {
    prob = RandUniform();
    prob_sum += prob;
    selection_probs.push_back(prob);
  }
  for (int i = 0; i < n; i++) {
    selection_probs[i] /= prob_sum;
  }

  // generate a outputs_set with ones and with size < k
  int ones_size = rand() % (n / 2) + 1;
  std::set<int> outputs_set;
  for (int i = 0; i < ones_size; i++) {
    outputs_set.insert(rand() % n);
  }

  // generate a random number k from 1 to n
  int k = 2 * ones_size;
  NormalizeVec(k, outputs_set, &selection_probs);

  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }

  int N = iters;
  Timer t;
  t.Reset();
  BaseFloat total_time;
  for (int i = 0; i < N; i++) {
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
  }
  total_time = t.Elapsed();
  KALDI_LOG << "Total time of the samping code " << N << " times is " << total_time;
}

}  // end namespace rnnlm
}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  using namespace rnnlm;
  int N = 10000;
  UnitTestSampleWithProbOne(N);
  UnitTestSamplingTime(N);
  UnitTestSampleConvergence();
}

