// online2/online2-nnet3-latgen-i2x-wrapper.cc

#include <cstdint>
#include <cstddef>
#include <lat/lattice-functions.h>

#include "nnet3/nnet-utils.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online2-nnet3-latgen-i2x-wrapper.h"

namespace kaldi {

class Decoder {

 public:
  Decoder(
      const TransitionModel &trans_model,
      const fst::Fst<fst::StdArc> &decode_fst,
      const LatticeFasterDecoderConfig &decoder_opts,
      const OnlineNnet2FeaturePipelineInfo &feature_info,
      const nnet3::DecodableNnetSimpleLoopedInfo &decodable_info,
      const fst::SymbolTable &word_syms,
      const OnlineIvectorExtractorAdaptationState adaptation_state, // TODO: do we really need it?
      BaseFloat samp_freq,
      int32 chunk_length) :
      word_syms_(word_syms),
      chunk_length_(chunk_length),
      float_sample_buffer_(chunk_length),
      samp_freq_(samp_freq) {

    feature_pipeline_ = new OnlineNnet2FeaturePipeline(feature_info);
    feature_pipeline_->SetAdaptationState(adaptation_state);
    decoder_ = new SingleUtteranceNnet3Decoder(
        decoder_opts, trans_model, decodable_info, decode_fst, feature_pipeline_);
    buffer_.reserve(2 * chunk_length_);

  }
  int32 GetFinalResult(std::string *result);
  int32 FeedChunk(int16 *data, size_t length);
  ~Decoder() {
    delete feature_pipeline_;
    delete decoder_;
  }
  bool IsFinalized() const { return finalized_; }
 private:
  SingleUtteranceNnet3Decoder *decoder_ = nullptr;
  OnlineNnet2FeaturePipeline *feature_pipeline_ = nullptr;
  const fst::SymbolTable& word_syms_;

  int32 chunk_length_;
  Vector<BaseFloat> float_sample_buffer_;
  BaseFloat samp_freq_;


  bool finalized_ = false;
  std::vector<int16> buffer_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(Decoder);

};

int32 Decoder::GetFinalResult(std::string *result) {
  if (!IsFinalized()) {
    KALDI_WARN << "Called GetFinalResult() on the un-finalized Decoder.\n"
               << "We will finalize it by force, but the correct way is\n"
               << "to call FeedChunk() with length == 0.\n"
               << "It is strongly advised to change your client accordingly.";
    FeedChunk(nullptr, 0); // force-finalize if the user failed to do so
  }

  CompactLattice clat;
  KALDI_ASSERT(finalized_ == true);
  decoder_->GetLattice(finalized_, &clat);

  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return -1;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  KALDI_LOG << "Likelihood per frame for utterance is "
            << (likelihood / num_frames) << " over " << num_frames
            << " frames.";

  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms_.Find(words[i]);
    if (s == "") {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table."
          << "The ASR resources are inconsistent!"
          << "Check that the output symbols are correct.";
      return -1;
    }
    if (i + 1 != words.size()) {
      s += " ";
    }
    *result += s;
  }
  return 0;
}

int32 Decoder::FeedChunk(int16_t *data, size_t length) {
  if (finalized_) {
    KALDI_WARN << "Calling FeedChunk with length == 0 for the second time (or more).\n"
               << "The decoder was already finalized!\n"
               << "The call is ignored. Create a new Decoder and work with it.";
    return -1;
  }
  for (size_t i = 0; i < length; i++) {
    buffer_.push_back(data[i]);
  }
  bool last_call = (length == 0);
  if (last_call) {
    finalized_ = true;
  }
  int32_t effective_chunk_length = last_call ? buffer_.size() : chunk_length_;
  if (buffer_.size() < effective_chunk_length) {
    return 0;
  }
  size_t leftover = (effective_chunk_length == 0) ? 0 : buffer_.size() % effective_chunk_length;
  size_t end = buffer_.size() - leftover;
  if (last_call) {
    float_sample_buffer_.Resize(effective_chunk_length);
  }
  for (size_t i = 0; i < end; i += effective_chunk_length) {
    for (size_t j = 0; j < effective_chunk_length; j++) {
      float_sample_buffer_(j) = buffer_[i + j];
    }
    feature_pipeline_->AcceptWaveform(samp_freq_, float_sample_buffer_);
  }
  if (last_call) {
    feature_pipeline_->InputFinished();
  }
  decoder_->AdvanceDecoding();
  if (last_call) {
    decoder_->FinalizeDecoding();
  }

  // now move the leftover samples to the beginning of the queue
  for (size_t i = 0; i < leftover; i++) {
    buffer_[i] = buffer_[end + i];
  }
  buffer_.resize(leftover);
  return 0;
}

class DecoderFactory {
 public:
  DecoderFactory(const std::string &);
  ~DecoderFactory();
  Decoder *StartDecodingSession() const;
 private:
  TransitionModel trans_model_;
  nnet3::AmNnetSimple am_nnet_;
  // feature_opts includes configuration for the iVector adaptation,
  // as well as the basic features.
  OnlineNnet2FeaturePipelineConfig feature_opts_;
  nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;
  LatticeFasterDecoderConfig decoder_opts_;
  OnlineEndpointConfig endpoint_opts_;

  int32 chunk_length_;
  bool do_endpointing_ = false;
  BaseFloat samp_freq_;

  OnlineNnet2FeaturePipelineInfo *feature_info_ = nullptr;
  nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_ = nullptr;
  fst::Fst<fst::StdArc> *decode_fst_ = nullptr;
  fst::SymbolTable *word_syms_ = nullptr;
  OnlineIvectorExtractorAdaptationState *adaptation_state_ = nullptr;
};

DecoderFactory::~DecoderFactory() {
  delete feature_info_;
  delete decodable_info_;
  delete decode_fst_;
  delete word_syms_;
  delete adaptation_state_;
}

DecoderFactory::DecoderFactory(const std::string &resource_dir_prefix) {
  using namespace fst;

  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;

  const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
          "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
          "optional endpointing.  Note: some configuration values and inputs are\n"
          "set via config files whose filenames are passed as options\n"
          "\n"
          "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
          "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
          "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
          "you want to decode utterance by utterance.\n";

  const std::string word_syms_rxfilename = resource_dir_prefix + "/words.txt";

  ParseOptions po(usage);
  BaseFloat chunk_length_secs = 0.18;
  po.Register("chunk-length", &chunk_length_secs,
              "Length of chunk size in seconds, that we process.  Set to <= 0 "
                  "to use all input in one chunk.");
  po.Register("do-endpointing", &do_endpointing_,
              "If true, apply endpoint detection");

  feature_opts_.Register(&po);
  decodable_opts_.Register(&po);
  decoder_opts_.Register(&po);
  endpoint_opts_.Register(&po);

  std::vector<std::string> strargs = {
      {"DUMMY"},
      {("--config=" + resource_dir_prefix + "/general.conf")},
      {("--mfcc-config=" + resource_dir_prefix + "/mfcc_hires.conf")},
      {(resource_dir_prefix + "/final.mdl")},
      {(resource_dir_prefix + "/HCLG.fst")}
  };

  size_t argc = strargs.size();
  char **argv = (char **) malloc(argc * sizeof(char *));
  for (size_t arg = 0; arg < argc; arg++) {
    std::string cur_arg = strargs[arg];
    argv[arg] = (char *) malloc((cur_arg.size() + 1) * sizeof(char));
    strcpy(argv[arg], cur_arg.c_str());
  }
  const char *argv_c[] = {argv[0], argv[1], argv[2], argv[3], argv[4]};
  KALDI_ASSERT(sizeof(argv_c) == argc * sizeof(char*));
  po.Read((int) argc, argv_c);

  if (po.NumArgs() != 2) {
    po.PrintUsage();
    KALDI_ERR << "Initialization error. " << po.NumArgs() << " != 2";
  }

  std::string nnet3_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2);

  feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_opts_);
  if (feature_info_->feature_type == "mfcc") {
    samp_freq_ = feature_info_->mfcc_opts.frame_opts.samp_freq;
  } else if (feature_info_->feature_type == "fbank") {
    samp_freq_ = feature_info_->fbank_opts.frame_opts.samp_freq;
  } else if (feature_info_->feature_type == "plp") {
    samp_freq_ = feature_info_->plp_opts.frame_opts.samp_freq;
  } else {
    KALDI_ERR << "feature_type should be mfcc, fbank or plp";
  }

  chunk_length_ = static_cast<int32> (samp_freq_ * chunk_length_secs);

  {
    bool binary;
    Input ki(nnet3_rxfilename, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_nnet_.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));
  }

  // this object contains precomputed stuff that is used by all decodable
  // objects.  It takes a pointer to am_nnet because if it has iVectors it has
  // to modify the nnet to accept iVectors at intervals.
  decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                             &am_nnet_);
  decode_fst_ = ReadFstKaldiGeneric(fst_rxfilename);

  if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
              << word_syms_rxfilename;
  }

  adaptation_state_ = new OnlineIvectorExtractorAdaptationState(
      feature_info_->ivector_extractor_info);

}

Decoder *DecoderFactory::StartDecodingSession() const {
  return new Decoder(
      trans_model_,
      *decode_fst_,
      decoder_opts_,
      *feature_info_,
      *decodable_info_,
      *word_syms_,
      *adaptation_state_,
      samp_freq_,
      chunk_length_);
}

} // namespace kaldi
using kaldi::DecoderFactory;
using kaldi::Decoder;

DecoderFactory *InitDecoderFactory(const std::string &resource_dir) {
  try {
    return new DecoderFactory(resource_dir);
  } catch (...) {
    return nullptr;
  }
}

Decoder *StartDecodingSession(const DecoderFactory *decoder_factory) {
  try {
    return decoder_factory->StartDecodingSession();
  } catch (...) {
    return nullptr;
  }
}
int32_t FeedChunk(Decoder *decoder, int16_t *data, size_t length) {
  return decoder->FeedChunk(data, length);
}

int32_t GetResultAndFinalize(Decoder *decoder, std::string *result) {
  return decoder->GetFinalResult(result);
}
