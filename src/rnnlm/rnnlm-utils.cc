#include "rnnlm/rnnlm-utils.h"

namespace kaldi {
namespace rnnlm {

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

vector<int> Select(const vector<BaseFloat> &u, int n) {
  vector<int> ans(n);

  BaseFloat tot_weight = 0;

  for (int i = 0; i < n; i++) {
    tot_weight += u[i];
    ans[i] = i;
  }
//  PrintVec(ans);

  for (int k = n; k < u.size(); k++) {
    tot_weight += u[k];
//    cout << "  tot-weight = " << tot_weight << endl;
    BaseFloat pi_k1_k1 = u[k] / tot_weight * n;

    if (pi_k1_k1 > 1) {
      pi_k1_k1 = 1;
    }

    BaseFloat p = BaseFloat(rand()) / RAND_MAX;
    if (p > pi_k1_k1) {
//      cout << "  not replace" << endl;
      continue;
    }

    vector<BaseFloat> R(n);
    // fill up R
    {
      BaseFloat Lk = 0;
      BaseFloat Tk = 0;
      for (int i = 0; i < n; i++) {
        BaseFloat pi_k_i = u[ans[i]] / (tot_weight - u[k]) * n;
        BaseFloat pi_k1_i = u[ans[i]] / tot_weight * n;

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
        BaseFloat pi_k_i = u[ans[i]] / (tot_weight - u[k]) * n;
        BaseFloat pi_k1_i = u[ans[i]] / tot_weight * n;

        if (pi_k_i < 1 && pi_k1_i < 1) {
          // case C
          R[i] = (1 - Tk) / (n - Lk);
//          cout << "C: R[" << i << "] is " << R[i] << endl;
        }
        sum += R[i];
      }
      assert(ApproxEqual(sum, 1.0));
    }
    p = BaseFloat(rand()) / RAND_MAX;

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
      ans[n - 1] = k;
    }
  }

  return ans;
}

} // namespace nnet3
} // namespace kaldi
