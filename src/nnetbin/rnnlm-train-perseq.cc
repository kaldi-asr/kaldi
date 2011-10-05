// nnetbin/rnnlm-train-perseq.cc

// Copyright 2011  Karel Vesely

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet/nnet-rnnlm.h"
#include "nnet/nnet-rnnlmaux.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

#include <fstream>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Recurrent Neural Network Language Model training by stochastic gradient descent.\n"
        "Usage:  rnnlm-train-perseq [options] <model-in> <traindata-rtxt> <dictionary-rtxt> [<model-out>]\n"
        "e.g.: \n"
        " rnnlm-train-perseq rnnlm.init train.txt dict rnnlm.iter1\n";

    ParseOptions po(usage);
    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.008,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    int32 min_seq_len = 128, bptt = 4;
    po.Register("min-seq-len", &min_seq_len, "Minimum length of sequence for one update");
    po.Register("bptt", &bptt, "Order of BPTT algorithm, number of steps back in time");

    bool preserve_state = true;
    po.Register("preserve-state", &preserve_state, "Preserve state of hidden layer across input sequences");

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        traindata_rtxt = po.GetArg(2),
        dict_rtxt = po.GetArg(3);
        
    std::string target_model_filename;
    if(!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

     
    using namespace kaldi;
    typedef kaldi::int32 int32;

    Rnnlm rnnlm;
    rnnlm.Read(model_filename);

    rnnlm.LearnRate(learn_rate);
    rnnlm.L2Penalty(l2_penalty);
    rnnlm.L1Penalty(l1_penalty);

    rnnlm.Bptt(bptt);
    rnnlm.PreserveState(preserve_state);

    kaldi::int64 tot_w = 0;

    //read the dictionary
    std::map<std::string,int32> dict;
    RnnlmAux::ReadDict(dict_rtxt,&dict);

    //open the training data
    std::ifstream traindata(traindata_rtxt.c_str());
    if (!traindata.good()) {
      KALDI_ERR << "Cannot open training data: " << traindata_rtxt;
    }

    Xent xent;

    Matrix<BaseFloat> rnnlm_cls_out, rnnlm_out, glob_err, cls_err;
    std::vector<int32> input_seq;

    Timer tim;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_oov_error = 0;
    while (!traindata.eof()) {
      //read input sequence
      input_seq.clear();
      while(input_seq.size() < min_seq_len && !traindata.eof()) {
        if (!RnnlmAux::AddLine(traindata,dict,&input_seq)) {
          num_oov_error++;
        } else {
          if(num_done % 100000 == 0) std::cout << num_done << ", " << std::flush;
          num_done++;
        }
      }

      rnnlm.Propagate(input_seq,&rnnlm_out);
        
      //prepare target vector
      std::vector<int32> target(input_seq);
      target.erase(target.begin());

      //add one more dummy word as target
      target.push_back(1);

      xent.Eval(rnnlm_out,target,&glob_err);

      //set zero error for prediction of dummy word
      glob_err.Row(glob_err.NumRows()-1).SetZero();

      if(!crossvalidate) {
        rnnlm.Backpropagate(glob_err);
      }

      tot_w += input_seq.size();
    }

    //clean up
    traindata.close();

    if(!crossvalidate) {
      rnnlm.Write(target_model_filename,binary);
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << tim.Elapsed() << "s, word/s" << tot_w/tim.Elapsed(); 

    KALDI_LOG << "Done " << tot_w << " words, " 
              << num_done << " sentences, " 
              << num_oov_error << " sentences with OOV words skipped.";

    KALDI_LOG << xent.Report();

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
