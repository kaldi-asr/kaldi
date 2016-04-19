// nnet/nnet-input-offset.h

#ifndef KALDI_NNET_NNET_INPUT_OFFSET_H_
#define KALDI_NNET_NNET_INPUT_OFFSET_H_

#include "nnet/nnet-nnet.h"

namespace kaldi {
namespace nnet1 {

class ParallelOneNnet {  // single nnet
 friend class Nnet;
 public:
  ParallelOneNnet(const bool freeze_update, const bool has_utt2spk, Nnet *nnet,
                  NnetDataRandomizerOptions *rnd_opts ) : freeze_update_(freeze_update),
                                                          has_utt2spk_(has_utt2spk),
                                                          mNnet_(nnet) {
    feed_forward_ = false;
    if(rnd_opts != NULL)
      feature_randomizer_.Init(*rnd_opts); 
    else
      feed_forward_ = true;
  }
  void InitFeatReader(const std::string &feature_rspecifier) {
    if(parallel_valid_string(feature_rspecifier)) {
      if(has_utt2spk_) 
        vec_reader_.Open(feature_rspecifier);
      else
        feature_reader_.Open(feature_rspecifier);
    }
  }
  void InitUtt2SpkReader(const std::string &utt2spk_filename) {
    if(parallel_valid_string(utt2spk_filename)) {
      utt2spk_reader_.Open(utt2spk_filename); 
    }
  }
  void InitFeatTransform(const std::string &feature_transform) {
    if(parallel_valid_string(feature_transform)) {
      nnet_transf_.Read(feature_transform);
    }
  }
  void InitNnet(const std::string &model_filename) {
    nnet_.Read(model_filename);
    nnet_.SetTrainOptions(mNnet_->GetTrainOptions());
    nnet_.opts_.freeze_update = freeze_update_;  // overwrite the following two variables
    nnet_.opts_.parallel_level = -1;
  }
  bool HasUtt(const std::string &utt) {
    if(has_utt2spk_) {
      return utt2spk_reader_.HasKey(utt);
    }
    return feature_reader_.HasKey(utt);
  }
  bool AddData(const std::string &utt, const int32 &nRows, const int32 time_shift = 0) {
    Matrix<BaseFloat> mat;
    if(has_utt2spk_) {
      std::string spk = utt2spk_reader_.Value(utt);
      Vector<BaseFloat> vec;
      vec = vec_reader_.Value(spk);
      mat.Resize(nRows, vec.Dim());
      mat.CopyRowsFromVec(vec);
    } else {
      mat = feature_reader_.Value(utt);
      if (mat.NumRows() < nRows)
        return false;
      else if(mat.NumRows() > nRows) {
        mat.Resize(nRows, mat.NumCols(), kCopyData);
      }
    }
    if(time_shift > 0) {
      int32 last_row = mat.NumRows() - 1; // last row,
      mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
      for (int32 r = last_row+1; r<mat.NumRows(); r++) {
        mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
      }
    }
    nnet_transf_.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf_);
    if(!feed_forward_)
      feature_randomizer_.AddData(feats_transf_);
    return true;
  }
  void FeatRandomize(const std::vector<int32> &mask) {
    feature_randomizer_.Randomize(mask);
  }
  void Propagate() {
    const CuMatrixBase<BaseFloat>& nnet_in = feed_forward_ ? feats_transf_ : feature_randomizer_.Value();
    nnet_.Propagate(nnet_in, &nnet_out_);
    if(!feed_forward_)
      feature_randomizer_.Next();
  }
  void Feedforward() {
    nnet_.Feedforward(feats_transf_, &nnet_out_);
  }
  void Write(const std::string &target_model_name, const bool binary) {
    nnet_.Write(target_model_name, binary);
  }
  static bool parallel_valid_string(const std::string &str) {
    if(str.empty())
      return false;
    size_t startpos = str.find_last_not_of(" \t");
    if(startpos == std::string::npos)
      return false;
    return true;
  }
 private:
  bool freeze_update_;
  RandomAccessBaseFloatMatrixReader  feature_reader_;
  RandomAccessBaseFloatVectorReader  vec_reader_;
  bool has_utt2spk_;
  RandomAccessTokenReader utt2spk_reader_;
  MatrixRandomizer feature_randomizer_;
  CuMatrix<BaseFloat> feats_transf_, nnet_out_;
  Nnet nnet_transf_;
  Nnet nnet_;
  Nnet *mNnet_;      // main nnet to which the current subnet attaches
  bool feed_forward_;
};

class ParallelNnet {
  friend class Nnet;
 public:
  ParallelNnet(NnetDataRandomizerOptions *rnd_opts, Nnet *nnet) : rnd_opts_(rnd_opts),
                                                                  mNnet_(nnet)
                                                                  { }
  ParallelNnet(Nnet *nnet) : rnd_opts_(NULL),
                             mNnet_(nnet)
                             { }
  ~ParallelNnet() {
    Destroy();
  }
  void SetParallelOpts(ParallelNnetOptions &opts) {
    opts_ = opts;
    CheckUpdate();
  }
  void Init() {
    for(int32 i = 0; i < vec_feature_.size(); ++i) {
      bool has_utt2spk = false;
      if(ParallelOneNnet::parallel_valid_string(vec_utt2spk_[i])) has_utt2spk =true;
      ParallelOneNnet *pOneNnet = new ParallelOneNnet(opts_.parallel_freeze_update, has_utt2spk, mNnet_, rnd_opts_);
      pOneNnet->InitNnet(vec_nnetfile_[i]);
      pOneNnet->InitFeatReader(vec_feature_[i]);
      pOneNnet->InitFeatTransform(vec_feature_transform_[i]);
      pOneNnet->InitUtt2SpkReader(vec_utt2spk_[i]);
      nnet_vec_.push_back(pOneNnet);
    }
    mNnet_->SetParallelNnet(&nnet_vec_, opts_.parallel_nnet_level);
  }
  bool HasUtt(const std::string &utt) {
    for(int32 i = 0; i < nnet_vec_.size(); ++ i) {
      if(! nnet_vec_[i]->HasUtt(utt))
        return false;
    }
    return true;
  }
  bool AddData(const std::string &utt, const int32 &nRows, const int32 time_shift = 0) {
    for(int32 i = 0; i < nnet_vec_.size(); ++i) {
      if(! nnet_vec_[i]->AddData(utt, nRows, time_shift))
        return false;
    }
    return true;
  }
  void FeatRandomize(const std::vector<int32> &mask) {
    for(int32 i = 0; i < nnet_vec_.size(); ++i) {
      nnet_vec_[i]->FeatRandomize(mask);
    }
  }
  void Write(const bool binary) {
    for(int32 i = 0; i < nnet_vec_.size(); ++i)
      nnet_vec_[i]->Write(vec_updatefile_[i], binary);
  }
 private:
  ParallelNnetOptions opts_;
  std::vector<std::string> vec_feature_;
  std::vector<std::string> vec_utt2spk_;
  std::vector<std::string> vec_feature_transform_;
  std::vector<std::string> vec_nnetfile_;
  std::vector<std::string> vec_updatefile_;
  NnetDataRandomizerOptions *rnd_opts_;
  Nnet *mNnet_;
  std::vector<ParallelOneNnet*> nnet_vec_; 
  void CheckUpdate() {
    const char *delim = ";";
    if(!opts_.parallel_feature.empty()) {
      SplitStringToVector(opts_.parallel_feature, delim, false, &vec_feature_);
      for(int32 i = 0; i < vec_feature_.size(); ++i) {
        vec_utt2spk_.push_back("");
        vec_feature_transform_.push_back("");
      }
    }
    if (!opts_.parallel_utt2spk.empty()) {
      std::vector<std::string> vec_str;
      SplitStringToVector(opts_.parallel_utt2spk, delim, false, &vec_str);
      KALDI_ASSERT(vec_feature_.size() == vec_str.size());
      for(int32 i = 0; i < vec_str.size(); ++i)
        vec_utt2spk_[i] = vec_str[i];
    }
    if(!opts_.parallel_feature_transform.empty()) {
      std::vector<std::string> vec_str;
      SplitStringToVector(opts_.parallel_feature_transform, delim, false, &vec_str);
      KALDI_ASSERT(vec_feature_.size() == vec_str.size());
      for(int32 i = 0; i < vec_str.size(); ++i)
        vec_feature_transform_[i] = vec_str[i];
    }
    if (!opts_.parallel_net.empty()) {
      SplitStringToVector(opts_.parallel_net, delim, false, &vec_nnetfile_);
      KALDI_ASSERT(vec_feature_.size() == vec_nnetfile_.size());
    }
    if (!opts_.parallel_update.empty()) {
      SplitStringToVector(opts_.parallel_update, delim, false, &vec_updatefile_);
      KALDI_ASSERT(vec_feature_.size() == vec_updatefile_.size());
    }
  }
  void Destroy() {
    for(int32 i = 0; i < nnet_vec_.size(); ++i)
      delete nnet_vec_[i];
    nnet_vec_.clear();
  }
};

} // namespace nnet1
} // namespace kaldi

#endif
