// nnet/nnet-lhuc-sat.h

#ifndef KALDI_NNET_NNET_LHUC_SAT_H_
#define KALDI_NNET_NNET_LHUC_SAT_H_

#include "nnet/nnet-lhuc.h"
#include "nnet/nnet-nnet.h"
namespace kaldi {
namespace nnet1 {
class Nnet;
class LhucNnet {
 public:
  LhucNnet(float lhuc_const, float lhuc_lrate_coef, 
           Nnet *nnet, Component::ComponentType actType) : lhuc_const_(lhuc_const),
                                                           lhuc_learn_rate_coef_(lhuc_lrate_coef),
                                                           nnet_(nnet),
                                                           act_type_(actType),
                                                           lhuc_dim_(0) { }
  LhucNnet() : lhuc_const_(2.0),
               lhuc_learn_rate_coef_(1.0),
               nnet_(NULL),
               act_type_(Component::kSigmoid),
               lhuc_dim_(0){ }
  ~LhucNnet() { Destroy(); }
  void SetNnet(Nnet *nnet) { nnet_ = nnet; }
  void SetActType(Component::ComponentType actType) { act_type_ = actType; }
  void SetLhucConst(float lhuc_const) { lhuc_const_ = lhuc_const; }
  void SetLhucLrateCoef(float lhuc_lrate_coef) { lhuc_learn_rate_coef_ = lhuc_lrate_coef; }
  void Init() {
    KALDI_ASSERT(nnet_ != NULL && lhuc_const_ > 0 && lhuc_learn_rate_coef_ > 0);
    int32 num_comp = nnet_->NumComponents();
    int32 num_lhuc_comp = 0;
    for(int32 i = 0; i < num_comp; ++i) {
      const Component &comp = nnet_->GetComponent(i);
      if(comp.GetType() == act_type_) {
        if(lhuc_dim_ == 0) { 
          lhuc_dim_ = comp.OutputDim(); 
        }
        num_lhuc_comp ++;
        LhucComp *lhuc_comp =  new LhucComp(lhuc_dim_, lhuc_const_, lhuc_learn_rate_coef_);
        lhuc_comp->SetScaleVec(NULL);
        comp_vec_.push_back(lhuc_comp);
        Insert(&comp, lhuc_comp);
      }
    }
    KALDI_ASSERT(num_lhuc_comp > 0);
  }
  void Init(Matrix<BaseFloat> &mat) {
    KALDI_ASSERT(nnet_ != NULL && lhuc_const_ > 0 && lhuc_learn_rate_coef_ > 0);
    int32 num_comp = nnet_->NumComponents();
    int32 lhuc_num_comp_required = 0;
    for(int32 i = 0; i < num_comp; ++i) {
      const Component &comp = nnet_->GetComponent(i);
      if(comp.GetType() == act_type_) lhuc_num_comp_required ++;
    }
    if (mat.NumRows() != lhuc_num_comp_required) {
      KALDI_ERR << "Required LHUC component number is " << lhuc_num_comp_required << ", but"
                << " actual component number is " << mat.NumRows();
    }
    int32 lhuc_num_comp = 0;
    for(int32 i = 0; i < num_comp; ++i) {
      const Component &comp = nnet_->GetComponent(i);
      if(comp.GetType() == act_type_) {
        if(lhuc_dim_ == 0) {
          lhuc_dim_ = mat.NumCols();
          KALDI_ASSERT(lhuc_dim_ == comp.OutputDim());
        }
        Vector<BaseFloat> lhuc_vec(lhuc_dim_);
        SubVector<BaseFloat>tmp(lhuc_vec, 0, lhuc_dim_);
        tmp.CopyFromVec(mat.Row(lhuc_num_comp));
        LhucComp *lhuc_comp =  new LhucComp(lhuc_dim_, lhuc_const_, lhuc_learn_rate_coef_);
        lhuc_comp->SetScaleVec(&lhuc_vec);
        comp_vec_.push_back(lhuc_comp);
        Insert(&comp, lhuc_comp);
        lhuc_num_comp ++;
      }
    }
    KALDI_ASSERT(lhuc_num_comp == mat.NumRows()); 
  }
  void Update(Matrix<BaseFloat> *mat) {
     mat->Resize(comp_vec_.size(), lhuc_dim_);
    for(int32 i = 0; i < comp_vec_.size(); ++i) {
      SubVector<BaseFloat> v(mat->Row(i));
      comp_vec_[i]->GetLhucVec(&v);        
    }
  }
  void LhucPropagate(const Component *input_comp, const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out, bool *copy_done) {
    if(input_comp->GetType() != act_type_) {
      *copy_done = false;
      return;
    }
    *copy_done = true;
    it_ = comp_map_.find(input_comp), it_end_ = comp_map_.end();
    KALDI_ASSERT(it_ != it_end_);
    LhucComp *lhuc_comp = it_->second;
    out->Resize(in.NumRows(), in.NumCols());
    lhuc_comp->PropagateFnc(in, out);
  }
  void LhucBackpropagate(const Component *input_comp, const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                         const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    if(input_comp->GetType() != act_type_) {
      return;
    }
    LhucComp *lhuc_comp = Find(input_comp);
    lhuc_comp->BackpropagateFnc(in, out, out_diff, in_diff);
    lhuc_comp->Update(in, out_diff);
  }
 private:
  void Insert(const Component *main_comp, LhucComp *lhuc_comp) {
    it_ = comp_map_.find(main_comp); 
    it_end_ = comp_map_.end();
    KALDI_ASSERT(it_ == it_end_);
   comp_map_.insert(std::pair<const Component*, LhucComp*>(main_comp, lhuc_comp));
  }
  LhucComp* Find(const Component *main_comp) {
    it_ = comp_map_.find(main_comp);
    it_end_ = comp_map_.end();
    KALDI_ASSERT(it_ != it_end_);
    return it_->second;
  }
  void Destroy() {
    it_ = comp_map_.begin(), it_end_ = comp_map_.end(); 
    for(; it_ != it_end_; ++ it_) {
      Component *lhuc_comp = it_->second;
      delete lhuc_comp;
    }
    comp_map_.clear();
  }
 private:
  float lhuc_const_;
  float lhuc_learn_rate_coef_;
  Nnet *nnet_;
  Component::ComponentType act_type_;
  int32 lhuc_dim_;
  std::vector<LhucComp*> comp_vec_;
  std::map<const Component*, LhucComp*> comp_map_;
  std::map<const Component*, LhucComp*>::iterator it_, it_end_;
};

class LhucSat{
 public:
  LhucSat(float lhuc_const, float lhuc_lrcoef,
          std::string &act_type, Nnet *nnet) : lhuc_const_(lhuc_const),
                                        lhuc_learn_rate_coef_(lhuc_lrcoef),
                                        act_type_(Component::MarkerToType(act_type)), model_wfilename_(""),
                                        nnet_(nnet),
                                        active_lhuc_nnet_(NULL) { }
  LhucSat() : lhuc_const_(2.0),
              lhuc_learn_rate_coef_(1.0),
              act_type_(Component::kSigmoid), model_wfilename_(""),
              nnet_(NULL),
              active_lhuc_nnet_(NULL) { }
  
  ~LhucSat() { Destroy(); }
  void SetLhucConst(float lhuc_const) { lhuc_const_ = lhuc_const; }
  void SetLrCoef(float lr_coef) { lhuc_learn_rate_coef_ = lr_coef; }
  void SetActType(std::string &act_type) { act_type_ = Component::MarkerToType(act_type); }
  
  void SetNnet(Nnet *nnet) {nnet_ = nnet; }
  LhucNnet* InsertLhuc(std::string &key, bool insert = false) {
    Matrix<BaseFloat> mat;
    return InitLhucTable(key, mat, insert);
  }
  /// we define lhuc_opts_csl as 
  /// lhuc_opts_csl=ark:utt2spk;input_model_filename,output_model_filename
  /// for instance, lhuc_opts_cls=ark:utt2spk;in_lhuc_sat.mdl,out_lhuc_sat.mdl
  void Init(const std::string &lhuc_opts_csl, Nnet *nnet) {
    std::vector<std::string> vec_component_str;
    SplitStringToVector(lhuc_opts_csl, ";", true, &vec_component_str);   
    KALDI_ASSERT(vec_component_str.size() == 2);
    std::vector<std::string> vec_str;
    SplitStringToVector(vec_component_str[1], ",", true, &vec_str);
    OpenUtt2Spk(vec_str[0]);
    KALDI_ASSERT(vec_str.size() >= 1);
    model_wfilename_ = "";
    if(vec_str.size() == 2)
      model_wfilename_ = vec_str[1];
    Read(vec_str[0]);
    nnet_ = nnet;
  }
  bool Utt2SpkLhuc(const std::string &utt) {
    if(!utt2spk_reader_.HasKey(utt))
      return false;
    std::string spk = utt2spk_reader_.Value(utt);
    if(InsertLhuc(spk, false) == NULL)
      return false;
    return true;
  }
  LhucNnet* GetActiveLhuc() { return active_lhuc_nnet_; }
  void Read(std::string &file) {
    bool binary;
    Input in(file, &binary);
    std::istream &is = in.Stream();
    KALDI_ASSERT(is.good());
    ExpectToken(is, binary, "<LhucConst>");
    ReadBasicType(is, binary, &lhuc_const_);
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &lhuc_learn_rate_coef_);
    ExpectToken(is, binary, "<NetActType>");
    std::string act_type;
    ReadToken(is, binary, &act_type);
    act_type_ = Component::MarkerToType(act_type);
    ReadTable(is, binary);
    in.Close();
    if(map_spk2nnet_.size() == 0) {
      KALDI_ERR << "Lhuc adaptive file '" << file << "' is empty.";
    }   
  }
  void Write(std::string &file, bool binary) {
    Output output(file, binary);
    std::ostream &os = output.Stream();
    WriteToken(os, binary, "<LhucConst>");
    WriteBasicType(os, binary, lhuc_const_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, lhuc_learn_rate_coef_);
    WriteToken(os, binary, "<NetActType>");
    std::string act_type = Component::TypeToMarker(act_type_);
    WriteToken(os, binary, act_type);
    it_ = map_spk2nnet_.begin(); 
    it_end_ = map_spk2nnet_.end();
    for(; it_ != it_end_; ++ it_) {
      WriteToken(os, binary, it_->first);
      LhucNnet *lhuc_nnet = it_->second;
      Matrix<BaseFloat> mat;
      lhuc_nnet->Update(&mat);
      mat.Write(os, binary);
    }   
    output.Close();
  }
  void Write(bool binary) {
    if(model_wfilename_ != "")
    Write(model_wfilename_, binary);
  }
  void OpenUtt2Spk(const std::string &utt2spk_rspecifier) {
    utt2spk_reader_.Open(utt2spk_rspecifier);
  }
 private:
  void ReadTable(std::istream &is, bool binary) {
    while(!is.eof()) {
      is.clear();
      std::string key;
      is >> key;
      if(key.empty()) break;
      Matrix<BaseFloat> mat;
      mat.Read(is, binary);
      InitLhucTable(key, mat, true);
    }
  } 
  LhucNnet* InitLhucTable(std::string &key, Matrix<BaseFloat> &mat, bool insert = true) {
     it_ = map_spk2nnet_.find(key);
     it_end_ = map_spk2nnet_.end();
      if(it_ != it_end_) {
        active_lhuc_nnet_ = it_->second;
        return active_lhuc_nnet_;
      }
      if(!insert)
        return NULL; 
      LhucNnet *lhuc_nnet = new LhucNnet(lhuc_const_, lhuc_learn_rate_coef_, nnet_, act_type_);
      if(mat.NumRows() != 0)
        lhuc_nnet->Init(mat);
      else
        lhuc_nnet->Init();
      map_spk2nnet_.insert(std::pair<std::string, LhucNnet*>(key, lhuc_nnet));
      active_lhuc_nnet_ = lhuc_nnet;
      return active_lhuc_nnet_;
    }
    void Destroy() {
    it_ = map_spk2nnet_.begin(); it_end_ = map_spk2nnet_.end();
    for(; it_ != it_end_; ++ it_) {
      LhucNnet *lhuc_nnet = it_->second;
      delete lhuc_nnet;
    }
  }
 private:
  float lhuc_const_;
  float lhuc_learn_rate_coef_;
  Component::ComponentType act_type_;
  RandomAccessTokenReader utt2spk_reader_;
  std::string model_wfilename_;
  Nnet *nnet_;
  LhucNnet *active_lhuc_nnet_;
  std::map<std::string, LhucNnet*> map_spk2nnet_;
  std::map<std::string, LhucNnet*>::iterator it_, it_end_;
};
}  // namespace nnet1
}  // namespace kaldi
#endif
