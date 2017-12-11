// online2/online2-nnet3-latgen-i2x-wrapper.cc

#include <cstdint>
#include <cstddef>
#include <exception>

#include <lat/lattice-functions.h>
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online2-nnet3-latgen-i2x-wrapper.h"

namespace kaldi {

class DecoderImpl {

 public:
  DecoderImpl(
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
    sample_buffer_.reserve(2 * chunk_length_);

  }
  const RecognitionResult GetResult();
  int32 FeedBytestring(const std::string& bytestring);
  int32 Finalize();

  ~DecoderImpl() {
    KALDI_LOG << "DecoderImpl::~DecoderImpl() -- inner destructor called. All internal structures cleared.";
    delete feature_pipeline_;
    delete decoder_;
    feature_pipeline_ = nullptr;
    decoder_ = nullptr;
  }
  bool IsFinalized() const { return finalized_; }
 private:
  int32 FeedChunk(const int16 *data, size_t length);

  SingleUtteranceNnet3Decoder *decoder_ = nullptr;
  OnlineNnet2FeaturePipeline *feature_pipeline_ = nullptr;
  const fst::SymbolTable& word_syms_;

  int32 chunk_length_;
  Vector<BaseFloat> float_sample_buffer_;
  BaseFloat samp_freq_;

  bool finalized_ = false;
  std::vector<int16> sample_buffer_;
  std::string byte_buffer_;
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecoderImpl);
};

int32 DecoderImpl::FeedBytestring(const std::string& bytestring) {
  byte_buffer_ += bytestring;
  bool leftover = (byte_buffer_.size() % 2 != 0);
  size_t length = leftover ? byte_buffer_.size() - 1 : byte_buffer_.size();
  length /= 2; // integer division: could become 0 if there was 1 byte in the buffer
  if (length == 0) {
    return 0; // never call FeedChunk with zero length if we don't want to finalize
  }
  int32_t return_code = FeedChunk(reinterpret_cast<const int16_t*>(byte_buffer_.c_str()), length);
  if (leftover) {
    char leftover_byte = byte_buffer_[byte_buffer_.size() - 1];
    byte_buffer_.clear();
    byte_buffer_.push_back(leftover_byte);
  } else {
    byte_buffer_.clear();
  }
  return return_code;
}

int32_t DecoderImpl::Finalize() {
  return FeedChunk(nullptr, 0); // a special call: finalize the decoder
}

const RecognitionResult DecoderImpl::GetResult() {
  RecognitionResult recognition_result;
  recognition_result.is_final = finalized_;

  if (decoder_->NumFramesDecoded() == 0) {
    return recognition_result;
  }
  
  CompactLattice clat;
  decoder_->GetLattice(finalized_, &clat);

  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice";
    recognition_result.error = true;
    return recognition_result;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  std::vector<int32> alignment;
  std::vector<int32> words;
  LatticeWeight weight;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  int32 num_frames = alignment.size();
  double likelihood = -(weight.Value1() + weight.Value2());
  KALDI_LOG << "Likelihood per frame for utterance is "
            << (likelihood / num_frames) << " over " << num_frames
            << " frames.";

  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms_.Find(words[i]);
    if (s == "") {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table."
          << "The ASR resources are inconsistent!"
          << "Check that the output symbols are correct.";
      recognition_result.error = true;
      return recognition_result;
    }
    if (i + 1 != words.size()) {
      s += " ";
    }
    recognition_result.transcript += s;
  }
  return recognition_result;
}

int32 DecoderImpl::FeedChunk(const int16_t *data, size_t length) {
  if (finalized_) {
    KALDI_WARN << "The decoder was already finalized!\n"
               << "The call is ignored. Create a new Decoder and work with it.";
    return -1;
  }
  for (size_t i = 0; i < length; i++) {
    sample_buffer_.push_back(data[i]);
  }
  bool last_call = (length == 0);
  if (last_call) {
    finalized_ = true;
  }
  int32_t effective_chunk_length = last_call ? sample_buffer_.size() : chunk_length_;
  if (sample_buffer_.size() < effective_chunk_length) {
    return 0;
  }
  size_t leftover = (effective_chunk_length == 0) ? 0 : sample_buffer_.size() % effective_chunk_length;
  size_t end = sample_buffer_.size() - leftover;
  if (last_call) {
    float_sample_buffer_.Resize(effective_chunk_length);
  }
  for (size_t i = 0; i < end; i += effective_chunk_length) {
    for (size_t j = 0; j < effective_chunk_length; j++) {
      float_sample_buffer_(j) = sample_buffer_[i + j];
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
    sample_buffer_[i] = sample_buffer_[end + i];
  }
  sample_buffer_.resize(leftover);

  return 0;
}

class DecoderFactoryImpl {
 public:
  DecoderFactoryImpl(const std::string &);
  ~DecoderFactoryImpl();
  DecoderImpl *StartDecodingSession() const;
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

DecoderFactoryImpl::~DecoderFactoryImpl() {
  delete feature_info_;
  delete decodable_info_;
  delete decode_fst_;
  delete word_syms_;
  delete adaptation_state_;

  feature_info_= nullptr;
  decodable_info_= nullptr;
  decode_fst_= nullptr;
  word_syms_= nullptr;
  adaptation_state_= nullptr;
}

DecoderFactoryImpl::DecoderFactoryImpl(const std::string &resource_dir_prefix) {
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
  for (size_t arg = 0; arg<argc; arg++) {
    std::string cur_arg = strargs[arg];
    argv[arg] = (char *) malloc((cur_arg.size() + 1) * sizeof(char));
    strcpy(argv[arg], cur_arg.c_str());
  }

  const char *argv_c[] = {argv[0], argv[1], argv[2], argv[3], argv[4]};
  KALDI_ASSERT(sizeof(argv_c) == argc * sizeof(char *));
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

  for (size_t arg = 0; arg < argc; arg++) {
    free(argv[arg]);
  }
  free(argv);
}

DecoderImpl *DecoderFactoryImpl::StartDecodingSession() const {
  return new DecoderImpl(
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
using kaldi::DecoderFactoryImpl;
using kaldi::DecoderImpl;

DecoderFactory::DecoderFactory(const std::string &resource_dir) :
  decoder_factory_impl_(new DecoderFactoryImpl(resource_dir)) {
  if (decoder_factory_impl_ == nullptr) {
    KALDI_WARN << "Decoder Factory creation did not succeed. "
	       << "Most likely caused by an out-of-memory error.";
  }
}

DecoderFactory::~DecoderFactory() {
  delete decoder_factory_impl_;
  decoder_factory_impl_ = nullptr;
}

Decoder* DecoderFactory::StartDecodingSession() {
  if (decoder_factory_impl_ == nullptr) {
    KALDI_WARN << "Failed to spawn a new Decoding Session. Decoder Factory is not valid. "
	       << "Most likely caused by an OOM when creating the decoder factory.";
    return nullptr;
  }
  try {
    return new Decoder(decoder_factory_impl_->StartDecodingSession());
  } catch (const std::exception& e) {
    KALDI_WARN << "Failed to spawn a new Decoding Session. " << e.what();
    return nullptr;
  }
}

Decoder::Decoder(DecoderImpl *decoder_impl) :
  decoder_impl_(decoder_impl)
{
  // Nothing to do.
}

Decoder::~Decoder() {
  KALDI_LOG << "Decoder::~Decoder() -- outer destructor called.";
  delete decoder_impl_;
  decoder_impl_ = nullptr;
}

int32_t Decoder::FeedBytestring(const std::string& bytestring) {
  if (decoder_impl_ == nullptr) {
    KALDI_WARN << "Decoder::FeedBytestring call failed. "
	       << "The Decoder was either already invalidated or not created properly";
    return -1;
  }
  return decoder_impl_->FeedBytestring(bytestring);
}

const RecognitionResult Decoder::GetResult() {
  if (decoder_impl_ == nullptr) {
    KALDI_WARN << "Decoder::GetResult call failed. "
	       << "The Decoder was either already invalidated or not created properly";
    RecognitionResult dummy;
    dummy.error = true;
    return dummy;
  }
  return decoder_impl_->GetResult();
}

int32_t Decoder::Finalize() {
  if (decoder_impl_ == nullptr) {
    KALDI_WARN << "Decoder::Finalize call failed. "
	       << "The Decoder was either already invalidated or not created properly";
    return -1;
  }
  return decoder_impl_->Finalize();
}

void Decoder::InvalidateAndFree() {
  delete decoder_impl_;
  decoder_impl_ = nullptr;
}
