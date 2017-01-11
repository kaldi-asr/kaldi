#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

bool LargerThan(const std::pair<int, BaseFloat> &t1,
                const std::pair<int, BaseFloat> &t2) {
  return t1.first > t2.first;
}

vector<string> SplitByWhiteSpace(const string &line) {
  std::stringstream ss(line);
  vector<string> ans;
  string word;
  while (ss >> word) {
    ans.push_back(word);
  }
  return ans;
}

unordered_map<string, int> ReadWordlist(string filename) {
  unordered_map<string, int> ans;
  ifstream ifile(filename.c_str());
  string word;
  int id;

  while (ifile >> word >> id) {
    ans[word] = id;
  }
  return ans;
}

void ReadUnigram(string f, vector<BaseFloat> *u) {
  ifstream ifile(f.c_str());
  int id;
  BaseFloat prob;
  BaseFloat sum = 0.0;
  while (ifile >> id >> prob) {
    KALDI_ASSERT(id == u->size());
    (*u).push_back(prob);
    sum += prob;
  }

  for (int i = 0; i < u->size(); i++) {
    (*u)[i] /= sum;
//    (*u)[i] = 1.0 / u->size();  // TODO(hxu)
  }
}

NnetExample GetEgsFromSent(const vector<int>& word_ids_in, int input_dim,
                           const vector<int>& word_ids_out, int output_dim) {
  SparseMatrix<BaseFloat> input_frames(word_ids_in.size(), input_dim);

  for (int j = 0; j < word_ids_in.size(); j++) {
    vector<std::pair<MatrixIndexT, BaseFloat> > pairs;
    pairs.push_back(std::make_pair(word_ids_in[j], 1.0));
    SparseVector<BaseFloat> v(input_dim, pairs);
    input_frames.SetRow(j, v);
  }

  NnetExample eg;
  eg.io.push_back(nnet3::NnetIo("input", 0, input_frames));

  Posterior posterior;
  for (int i = 0; i < word_ids_out.size(); i++) {
    vector<std::pair<int32, BaseFloat> > p;
    p.push_back(std::make_pair(word_ids_out[i], 1.0));
    posterior.push_back(p);
  }

  eg.io.push_back(nnet3::NnetIo("output", output_dim, 0, posterior));
  return eg;
}

void SampleWithoutReplacement(vector<std::pair<int, BaseFloat> > u, int n,
                              vector<int> *out) {
  sort(u.begin(), u.end(), LargerThan);

  KALDI_ASSERT(n != 0);

  vector<int>& ans = *out;
  ans.resize(n);

  BaseFloat tot_weight = 0;

  for (int i = 0; i < n; i++) {
    tot_weight += u[i].second;
    ans[i] = i;
  }
//  PrintVec(ans);

  for (int k = n; k < u.size(); k++) {
    tot_weight += u[k].second;
//    cout << "  tot-weight = " << tot_weight << endl;
    BaseFloat pi_k1_k1 = u[k].second / tot_weight * n;

    if (pi_k1_k1 > 1) {
      pi_k1_k1 = 1;  // must add
    } else {
      BaseFloat p = RandUniform();
      if (p > pi_k1_k1) {
  //      cout << "  not replace" << endl;
        continue;
      }
    }

    vector<BaseFloat> R(n);
    // fill up R
    {
      BaseFloat Lk = 0;
      BaseFloat Tk = 0;
      for (int i = 0; i < n; i++) {
        BaseFloat pi_k_i = u[ans[i]].second / (tot_weight - u[k].second) * n;
        BaseFloat pi_k1_i = u[ans[i]].second / tot_weight * n;

        // TODO(hxU) very strict test which actually isn't really neccesary
        if (u[ans[i]].second >= 1) {
          KALDI_ASSERT(pi_k_i >= 1 && pi_k1_i >= 1);
        }

//        cout << "piki, pik1i are " << pi_k_i << " " << pi_k1_i << endl;

        if (pi_k_i >= 1 && pi_k1_i >= 1) {
          // case A
          R[i] = 0;
//          cout << "A: R[" << i << "] is " << R[i] << endl;
          Lk++;
        } else if (pi_k_i >= 1 && pi_k1_i < 1) {
          // case B
          R[i] = (1 - pi_k1_i) / pi_k1_k1;
//          cout << "B: R[" << i << "] is " << R[i] << endl;
          Tk += R[i];
          Lk++;
        } else if (pi_k_i < 1 && pi_k1_i < 1) { // case C we will handle in another loop
        } else {
          assert(false);
        }
      }

      BaseFloat sum = 0;
      for (int i = 0; i < n; i++) {
        BaseFloat pi_k_i = u[ans[i]].second / (tot_weight - u[k].second) * n;
        BaseFloat pi_k1_i = u[ans[i]].second / tot_weight * n;

        if (pi_k_i < 1 && pi_k1_i < 1) {
          // case C
          R[i] = (1 - Tk) / (n - Lk);
//          cout << "C: R[" << i << "] is " << R[i] << endl;
        }
        sum += R[i];
      }
      assert(ApproxEqual(sum, 1.0));
    }
    BaseFloat p = RandUniform();

  //    cout << "  rand is " << p; 

    bool replaced = false;
    for (int i = 0; i < n; i++) {
      p -= R[i];
      if (p <= 0) {
        // i is the choice
  //        cout << "  chosen " << i << endl;
        ans[i] = k;
        replaced = true;
//        PrintVec(ans);
        break;
      }
    }

//    assert(replaced);
    if (!replaced) {
//      assert(p < 0.0000001);
      KALDI_LOG << "p should be close to 0; it is " << p;
      for (int i = 1; ; i++) {
        if (u[ans[n - i]].second < 1) {
          ans[n - 1] = k;
          break;
        }
      }
    }
  }

  //  change to the correct indexes
  for (int i = 0; i < ans.size(); i++) {
    ans[i] = u[ans[i]].first;
  }
}

void NormalizeVec(int k, const set<int>& ones, vector<BaseFloat> *probs) {
  KALDI_ASSERT(ones.size() < k);
  // first check the unigrams add up to 1
  BaseFloat sum = 0;
  for (int i = 0; i < probs->size(); i++) {
    sum += (*probs)[i];
//    KALDI_LOG << sum;
  }
//  KALDI_LOG << "sum should be close to 1.0: " << sum;
  KALDI_ASSERT(ApproxEqual(sum, 1.0));
  
  // set the 1s
//  for (int i = 0; i < ones.size(); i++) {
//    sum -= (*probs)[ones[i]];
//    (*probs)[ones[i]] = 10;  // make it way larger to avoid numerical issues
//  }
  for (set<int>::const_iterator iter = ones.begin(); iter != ones.end(); iter++) {
    sum -= (*probs)[*iter];
    (*probs)[*iter] = 10;  // mark the ones
  }

  // distribute the remaining probs
  BaseFloat maxx = 0.0;
  for (int i = 0; i < probs->size(); i++) {
    BaseFloat t = (*probs)[i];
    if (t < 2.0) {
      maxx = std::max(maxx, t);
    }
  }

//  KALDI_LOG << "sum should be smaller than 1.0: " << sum;
//  KALDI_LOG << "max is " << maxx;
//  KALDI_ASSERT(false);

  if (maxx * (k - ones.size()) * 1.0 / sum <= 1.0) {
//    KALDI_LOG << "no need to interpolate";
    for (int i = 0; i < probs->size(); i++) {
      BaseFloat &t = (*probs)[i];
      if (t > 2.0) {
        continue;
      }
      t = t * (k - ones.size()) * 1.0 / sum;
    }
  } else {
//    KALDI_LOG << "need to interpolate";
    // now we want to interpolate with a uniform prob of
    // (k - ones.size()) / (probs->size() - ones.size()) s.t. max is 1
    // w a + (1 - w) b = 1
    // ===> w = (1 - b) / (a - b)
    BaseFloat a = maxx * (k - ones.size()) * 1.0 / sum;
    BaseFloat b = (k - ones.size()) / (probs->size() - ones.size());
    KALDI_ASSERT(b <= 1);  // b == 1 usually happens when n = k
                           // in which case we must use a uniform distribution
    BaseFloat w = (1.0 - b) / (a - b);
    KALDI_ASSERT(w >= 0.0 && w <= 1.0);

    for (int i = 0; i < probs->size(); i++) {
      BaseFloat &t = (*probs)[i];
      if (t > 2.0) {
        continue;
      }
      t = w * t + (1.0 - w) * b;
    }
  }

  for (set<int>::const_iterator iter = ones.begin(); iter != ones.end(); iter++) {
    KALDI_ASSERT((*probs)[*iter] > 2.0);
    (*probs)[*iter] = 2.0;  // to avoid numerical issues
  }
  
  sum = 0.0;
  for (int i = 0; i < probs->size(); i++) {
    BaseFloat t = (*probs)[i];
    if (t > 2.0 || t <= 0) {
      KALDI_ASSERT(false);
    } 

    sum += std::min(BaseFloat(1.0), t);
    KALDI_LOG << "sum is " << sum;
  }
  KALDI_ASSERT(ApproxEqual(sum, k));

}

} // namespace rnnlm
} // namespace kaldi
