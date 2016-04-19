// nnet/nnet-mnet.h

#ifndef KALDI_NNET_MNET_H_
#define KALDI_NNET_MNET_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-matrix.h"
#include "nnet/nnet-nnet.h"

namespace kaldi {
namespace nnet1 {
class MultiNnet {
 public:
  MultiNnet():objective_function_("xent"),
              nnet_(NULL),
              total_lang_absence_(0),
              total_minibatch_(0) { }
  ~MultiNnet() {
    Destroy();
  }
  void Destroy() {
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      Nnet *pnnet = vec_nnet_[i];
      delete pnnet;
    }
  }
  void Init(const std::string &mling_opts_str,
            const NnetTrainOptions trn_opts,
            Nnet* nnet,  
            const NnetDataRandomizerOptions &rnd_opts,
            const std::string &objective_function) {
    std::vector<std::string> vec_component_str;
    SplitStringToVector(mling_opts_str, ";", true, &vec_component_str);   
    if(vec_component_str.size() < 3) {
      KALDI_ERR << "unexpected mling_opts_str " << "'" 
      << mling_opts_str << "'";
    }
    nnet_ = nnet;
    utt2lang_reader_.Open(vec_component_str[0]);
    std::vector<std::string> vec_lang_name;
    SplitStringToVector(vec_component_str[1], ",", true, &vec_lang_name);
    std::vector<std::string> vec_model_rfilename;
    SplitStringToVector(vec_component_str[2], ",", true, &vec_model_rfilename);
    KALDI_ASSERT(vec_lang_name.size() == vec_model_rfilename.size());
    FillMap(vec_lang_name);
    InitNnet(vec_model_rfilename, trn_opts);
    if(vec_component_str.size() > 3) {
      SplitStringToVector(vec_component_str[3], ",", true, &vec_model_wfilename_);
      KALDI_ASSERT(vec_model_wfilename_.size() == vec_model_rfilename.size());
    }
    lang_randomizer_.Init(rnd_opts);
    objective_function_ = objective_function;
    vec_nnet_.resize(vec_model_rfilename.size());
  }
  bool SetLangIdVec(const std::string &utt, int32 vec_dim) {
    if(!utt2lang_reader_.HasKey(utt)) {
      return false;
    }
    std::string lang_code = utt2lang_reader_.Value(utt); 
    std::map<const std::string, int32>::iterator it, it_end = map_lang2index_.end();
    int32 lang_id;
    it = map_lang2index_.find(lang_code);
    if(it == it_end)
      return false;
    lang_id = it->second;
    lang_id_vec_.Resize(vec_dim);
    lang_id_vec_.Set(static_cast<BaseFloat>(lang_id));
    return true;
  }
  void Resize(int32 vec_dim) {
    lang_id_vec_.Resize(vec_dim, kCopyData);
  }
  void AddData() {
    lang_randomizer_.AddData(lang_id_vec_);
  }
  void Randomize(const std::vector<int32> &mask) {
    lang_randomizer_.Randomize(mask); 
  }
  void GetData() {
    const Vector<BaseFloat> &lang_id_vec = lang_randomizer_.Value();
    lang_id_vec_.Resize(lang_id_vec.Dim());
    lang_id_vec_.CopyFromVec(lang_id_vec);
    lang_randomizer_.Next();  
  }
  void SetNnetTarget(const Posterior *nnet_tgt) {
    nnet_tgt_ = nnet_tgt;
  }
  void SetObjective(Xent *xent, Mse *mse) {
    xent_ = xent, mse_ = mse_;
  }
  // forward propagating
  void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ASSERT(lang_id_vec_.Dim() == in.NumRows());
    nnet_->propagate_buf_[0].Resize(in.NumRows(), in.NumCols());
    nnet_->propagate_buf_[0].CopyFromMat(in);
    int32 i = 0;
    for(; i < nnet_->components_.size() - 2; i++) {
      nnet_->components_[i]->Propagate(nnet_->propagate_buf_[i], &nnet_->propagate_buf_[i+1]);
    }
    // go through the last two layers
    int32 total_rows = 0, nrows = nnet_->propagate_buf_[i].NumRows(), ncols = nnet_->propagate_buf_[i].NumCols();
    vec_out_.resize(0); 
    vec_index_.resize(0);
    vec_tgt_.resize(0);
    for (int32 j=0; j< vec_nnet_.size(); j++) {
      int32 row_index = 0;
      Posterior post;
      CuMatrix<BaseFloat> tmp_mat(nrows, ncols);
      for(int32 k=0; k<lang_id_vec_.Dim(); k++) {
        int32 index_j = static_cast<int32>(lang_id_vec_(k));
        if(index_j == j) {   // the input belongs to the current language
          CuSubMatrix<BaseFloat> sub_mat(tmp_mat, row_index, 1, 0, ncols); 
          sub_mat.CopyFromMat(nnet_->propagate_buf_[i].Range(k, 1, 0, ncols));
          vec_index_.push_back(k);
          post.push_back((*nnet_tgt_)[k]);
          row_index ++; 
        }
      }
      vec_tgt_.push_back(post);
      if(row_index == 0) { // no input belongs to the current language
        CuMatrix<BaseFloat> out;
        vec_out_.push_back(out);   // an uninitialized matrix as output
        total_lang_absence_ ++;
        continue;
      }
      CuMatrix<BaseFloat> mling_in(row_index, ncols);
      mling_in.CopyFromMat(tmp_mat.Range(0, row_index, 0, ncols));
      Nnet *nnet = vec_nnet_[j];
      CuMatrix<BaseFloat> mling_out;
      nnet->Propagate(mling_in, &mling_out);
      KALDI_ASSERT(mling_in.NumRows() == mling_out.NumRows());
      vec_out_.push_back(mling_out);
      total_rows += row_index;
    }
    KALDI_ASSERT(total_rows == nrows);
    KALDI_ASSERT(vec_index_.size() == nrows && vec_tgt_.size() == vec_nnet_.size()); 
    total_minibatch_ ++;
    if(total_minibatch_ % 5000 == 0) {
      KALDI_VLOG(1) << "After " << total_minibatch_ << " minibatches, "
                    << "we have " << total_lang_absence_/total_minibatch_
                    << " language absence per minibatches.";
    }
  }
  void Eval(const Vector<BaseFloat> &frm_weights) {
    vec_diff_.resize(0);
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      CuMatrix<BaseFloat> obj_diff;
      if(vec_out_[i].NumRows() > 0) {
        Vector<BaseFloat> cur_frm_weights(SubVector<BaseFloat>(frm_weights, 0, vec_out_[i].NumRows()));
        if(objective_function_ == "xent") {
          xent_->Eval(cur_frm_weights, vec_out_[i], vec_tgt_[i], &obj_diff); 
        } else if(objective_function_ == "mse") {
          mse_->Eval(cur_frm_weights, vec_out_[i], vec_tgt_[i], &obj_diff);
        } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function_;
        }
      }
      vec_diff_.push_back(obj_diff);
    }
  }
  void Backpropagate(CuMatrix<BaseFloat> *in_diff) {
    KALDI_ASSERT(nnet_->NumComponents() != 0 && vec_diff_.size() == vec_nnet_.size());
    int32 num_rows = 0;
    for(int32 i = 0; i < vec_diff_.size(); i++) {
      num_rows += vec_diff_[i].NumRows();
    }
    KALDI_ASSERT(num_rows == vec_index_.size());
    const int32 &num_cols = vec_nnet_[0]->InputDim();
    CuMatrix<BaseFloat> out_diff(num_rows, num_cols, kSetZero);
    int32 total_row = 0;
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      if(vec_out_[i].NumRows() > 0) {
        CuMatrix<BaseFloat> cur_diff;
        vec_nnet_[i]->Backpropagate(vec_diff_[i], &cur_diff);
        for(int32 j = 0; j < cur_diff.NumRows(); ++j, ++total_row) {
          int32 cur_row = vec_index_[total_row];
          CuSubMatrix<BaseFloat> x(out_diff, cur_row, 1, 0, num_cols);
          x.CopyFromMat(cur_diff.Range(j, 1, 0, num_cols));
        }
      }
    }
    KALDI_ASSERT(total_row == num_rows);
    nnet_->backpropagate_buf_[nnet_->NumComponents()-2] = out_diff;
    // backpropagate using buffers
    for (int32 i = nnet_->NumComponents()-3; i >= 0; i--) {
      nnet_->components_[i]->Backpropagate(nnet_->propagate_buf_[i], nnet_->propagate_buf_[i+1],
                            nnet_->backpropagate_buf_[i+1], &nnet_->backpropagate_buf_[i]);
      if (nnet_->components_[i]->IsUpdatable()) {
        UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_->components_[i]);
        uc->Update(nnet_->propagate_buf_[i], nnet_->backpropagate_buf_[i+1]);
      }
    }
    // eventually export the derivative
    if (NULL != in_diff) (*in_diff) = nnet_->backpropagate_buf_[0];
  }
  void InfoGradient(std::ostringstream &ostr, int32 mIndex) {
    ostr << "Component " << mIndex+1 << " {" << std::endl;
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      ostr << "  Multilingual " << i+1 << " {" << std::endl;
      ostr << "  " << vec_nnet_[i]->InfoGradient();
      ostr << "  }" << std::endl; 
    }
    ostr << "}" << std::endl;
  }
  void InfoPropagate(std::ostringstream &ostr, int32 mIndex) {
    ostr << "[" << mIndex + 1 << "] output of {" << std::endl;
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      ostr << "  Multilingual [" << i+1 << "[ {" << std::endl;
      ostr << "  " << vec_nnet_[i]->InfoPropagate();
      ostr << "  }" << std::endl;
    }
    ostr << "}" <<std::endl;
  }
  void InfoBackPropagate(std::ostringstream &ostr, int32 mIndex) {
    ostr << "[" << mIndex + 1 << "] diff-output of {" << std::endl;
    for(int32 i = 0; i < vec_nnet_.size(); ++i) {
      ostr << "  Multilingual [" << i+1 << "[ {" << std::endl;
      ostr << "  " << vec_nnet_[i]->InfoBackPropagate();
      ostr << "  }" << std::endl;
    }
    ostr << "}" << std::endl;
  }
  void Write(bool binary) {
    for(int32 i = 0; i < vec_nnet_.size(); i++) {
      vec_nnet_[i]->Write(vec_model_wfilename_[i], binary);
    }
  }
 private:
  void FillMap(std::vector<std::string> &vec_lang_name) {
    for(int32 i = 0; i < vec_lang_name.size(); i++) {
      std::map<std::string, int32>::iterator it, it_end = map_lang2index_.end();
      it = map_lang2index_.find(vec_lang_name[i]);
      if(it != it_end) {
        KALDI_ERR << "lang code " << vec_lang_name[i] << "duplicated";
      }
      map_lang2index_.insert(std::pair<std::string, int32>(vec_lang_name[i], i));
    }
  }
  void InitNnet(std::vector<std::string> &vec_model_rfilename, 
                const NnetTrainOptions &trn_opts) {
    for(int32 i = 0; i < vec_model_rfilename.size(); i ++) {
      Nnet *pNnet = new Nnet();
      pNnet->Read(vec_model_rfilename[i]);
      pNnet->SetTrainOptions(trn_opts);
      vec_nnet_.push_back(pNnet);
    }
  }
 private:
  RandomAccessTokenReader utt2lang_reader_;
  std::vector<std::string> vec_model_wfilename_;
  std::map<const std::string,int32> map_lang2index_;
  std::string objective_function_;
  Vector<BaseFloat> lang_id_vec_;
  
  VectorRandomizer lang_randomizer_;
  Nnet *nnet_;
  std::vector<Nnet*> vec_nnet_;
  std::vector<CuMatrix<BaseFloat> > vec_out_;
  std::vector<int32> vec_index_;
  const Posterior *nnet_tgt_;
  std::vector<Posterior> vec_tgt_;
  std::vector<CuMatrix<BaseFloat> > vec_diff_;
  Xent *xent_;
  Mse  *mse_;
  int64 total_lang_absence_, total_minibatch_; // for log
};

}  // namespace nnet1
}  // namespace kaldi

#endif
