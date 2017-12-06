// online2/online2-nnet3-latgen-i2x-wrapper.cc

#include <cstdint>
#include <cstddef>

#include "base/kaldi-types.h"
#include "base/kaldi-utils.h"
#include "hmm/transition-model.h"
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
      const OnlineIvectorExtractorAdaptationState adaptation_state,
      BaseFloat samp_freq,
      int32 chunk_length) :
      chunk_length_(chunk_length),
      float_sample_buffer_(chunk_length),
      samp_freq_(samp_freq) {

    feature_pipeline_ = new OnlineNnet2FeaturePipeline(feature_info);
    feature_pipeline_->SetAdaptationState(adaptation_state);
    decoder_ = new SingleUtteranceNnet3Decoder(
        decoder_opts, trans_model, decodable_info, decode_fst, feature_pipeline_);
    buffer_.reserve(2 * chunk_length_);

  }
  int32_t FeedChunk(uint16_t *data, size_t length);
  ~Decoder() {
    delete feature_pipeline_;
    delete decoder_;
  }

 private:
  SingleUtteranceNnet3Decoder *decoder_ = nullptr;
  OnlineNnet2FeaturePipeline *feature_pipeline_ = nullptr;

  int32 chunk_length_;
  Vector<BaseFloat> float_sample_buffer_;
  BaseFloat samp_freq_;

  std::vector<uint16_t> buffer_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(Decoder);

};

int32_t Decoder::FeedChunk(uint16_t *data, size_t length) { // TODO implement last_call behavior

  for (size_t i = 0; i < length; i++) {
    buffer_.push_back(data[i]);
  }
  bool last_call = (length == 0);
  int32_t effective_chunk_length = last_call ? buffer_.size() : chunk_length_;
  if (buffer_.size() < effective_chunk_length) {
    return 0;
  }
  size_t leftover = buffer_.size() % effective_chunk_length;
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
  for (size_t i = 0; i < leftover; i++) {
    buffer_[i] = buffer_[end + i];
  }
  buffer_.resize(leftover);
  return 0;
}

class DecoderFactory {
 public:
  DecoderFactory(const char *);
  ~DecoderFactory();
  Decoder *StartDecodingSession() const;
 private:
  TransitionModel trans_model_;
  nnet3::AmNnetSimple am_nnet_;
  std::string word_syms_rxfilename_;
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

DecoderFactory::DecoderFactory(const char *resource_dir) {
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

  const std::string word_syms_rxfilename = std::string(resource_dir) + "/words.txt";

  ParseOptions po(usage);
  BaseFloat chunk_length_secs;
  po.Register("chunk-length", &chunk_length_secs,
              "Length of chunk size in seconds, that we process.  Set to <= 0 "
                  "to use all input in one chunk.");
  po.Register("do-endpointing", &do_endpointing_,
              "If true, apply endpoint detection");

  feature_opts_.Register(&po);
  decodable_opts_.Register(&po);
  decoder_opts_.Register(&po);
  endpoint_opts_.Register(&po);

  std::string resource_dir_prefix(resource_dir);

  std::vector<std::string> strargs = {
      {"DUMMY"},
      {(resource_dir_prefix + "/final.mdl")},
      {(resource_dir_prefix + "/HCLG.fst")},
      {("--config=" + resource_dir_prefix + "/general.conf")},
      {("--mfcc-conf=" + resource_dir_prefix + "/mfcc.conf")}
  };

  size_t argc = strargs.size();
  char** argv = (char**) malloc(argc * sizeof(char*));
  for (size_t arg = 0; arg < argc; arg++) {
    std::string cur_arg = strargs[arg];
    argv[arg] = (char*) malloc(cur_arg.size() * sizeof(char));
    strcpy(argv[arg], cur_arg.c_str());
  }
  const char* argv_c[] = {argv[0], argv[1], argv[2], argv[3], argv[4]};
  assert(sizeof(argv_c) == argc);
  po.Read((int)argc, argv_c);

  if (po.NumArgs() != 2) {
    po.PrintUsage();
    KALDI_ERR << "Initialization error.";
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
    *adaptation_state_,
    samp_freq_,
    chunk_length_);
}

} // namespace kaldi
using kaldi::DecoderFactory;
using kaldi::Decoder;
/*
   Creates a decoder factory.
   Called only once during the lifetime.
   mmaps extremely heavy resources like WFST and AM (up to 20 GB).
   Calling it more than once is likely to cause OOM!
   Returns a handle to a decoder factory,
   which will create light-weighted decoder objects (one per session).
   Returns nullptr on failure.
*/
DecoderFactory *InitDecoderFactory(const char *resource_dir) {
  return new DecoderFactory(resource_dir);
}

// Creates a decoder object.
// Returns nullptr on failure.
Decoder *StartDecodingSession(const DecoderFactory *decoder_factory) {
  return decoder_factory->StartDecodingSession();
}
// Feed PCM SI16 data into the decoder.
// Returns 0 on success, error code otherwise.
int32_t FeedChunk(Decoder *decoder, uint16_t *data, size_t length) {

  return decoder->FeedChunk(data, length);

}
/*
Gets current (ongoing) recognition result,
probably as a JSON or maybe protobuf
(with word timings and other stuff).
*/
std::string GetCurrentResult(const Decoder *);
// Frees the resources and destroys the recognition session.
// Returns 0 on success, error code otherwise.
int32_t Finalize(Decoder *);

