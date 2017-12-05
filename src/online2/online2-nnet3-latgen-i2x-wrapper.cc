// online2/online2-nnet3-latgen-i2x-wrapper.cc

#include <stdint.h>

#include "online2-nnet3-latgen-i2x-wrapper.cc"

namespace kaldi {

class Decoder {

public:
  Decoder(
    const TransitionModel& trans_model,
    const fst::Fst<fst::StdArc>& decode_fst,
    const LatticeFasterDecoderConfig& decoder_opts,
    const OnlineNnet2FeaturePipelineInfo& feature_info,
    const nnet3::DecodableNnetSimpleLoopedInfo& decodable_info,
    const AdaptationState adaptation_state,
    BaseFloat samp_freq,
    int32 chunk_length) : 
        chunk_length_(chunk_length), 
        float_sample_buffer_(chunk_length), 
        samp_freq_(samp_freq) {
   
    
    feature_pipeline_ = new OnlineNnet2FeaturePipeline(feature_info); 
    feature_pipeline->SetAdaptationState(adaptation_state);
    decoder_ = new SingleUtteranceNnet3Decoder(
                    decoder_opts, trans_model, decodable_info, decode_fst, feature_pipeline_);
    buffer_.reserve(2*chunk_length_);

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
  std::vector<uint16_t> buffer_;
  BaseFloat samp_freq_;
    
  Vector<BaseFloat> float_sample_buffer_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Decoder);

};

int32_t Decoder::FeedChunk(uint16_t *data, size_t length, bool last_call){ // TODO implement last_call behavior
   if (last_call && length > 0) {
        KALDI_ERR << "last_call should not add new data"; // TODO turn into assert or something
   }
   for (size_t i = 0; i < length; i++) { 
        buffer_.push_back(data[i]);
   }
   
   if (buffer_.size() < chunk_length_) {
        if (last_call && buffer_.size() > 0) {
            buffer_.resize(chunk_length_); // TODO find a better way to deal with ending? zero-padding might introduce bias
        }
        else {
            return 0;
        }
   }
   size_t leftover = buffer_.size() % chunk_length_;
   size_t end = buffer_.size() - leftover;
   for (size_t i = 0; i < end; i += chunk_length_) {
        for (size_t j = 0; j < chunk_length; j++) {
            float_sample_buffer_[j] = buffer_[i + j];
        }
        feature_pipeline_.AcceptWaveform(samp_freq_, float_sample_buffer_);
   }
   decoder.AdvanceDecoding();
   for (size_t i = 0; i < leftover; i++) {
        buffer_[i] = buffer_[end + i];
   }
   buffer_.resize(leftover);
}

class DecoderFactory {
public:
  DecoderFactory(const char*);
  ~DecoderFactory();
  Decoder* StartDecodingSession();
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

DecoderFactory::DecoderFactory(const char* resource_dir) {
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

    std::string string word_syms_rxfilename_;

    ParseOptions po(usage);
    BaseFloat chunk_length_secs;
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename_,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing_,
                "If true, apply endpoint detection");

    feature_opts_.Register(&po);
    decodable_opts_.Register(&po);
    decoder_opts_.Register(&po);
    endpoint_opts_.Register(&po);

    std::string resource_dir_prefix(resource_dir);

    const char* argv[] = {
        {"DUMMY"},
        {(resource_dir_prefix + "/final.mdl").c_str()},
        {resource_dir_prefix + "/HCLG.fst"},
        {"--config=" + resource_dir_prefix + "/general.conf"},
        {"--mfcc-conf=" + resource_dir_prefix + "/mfcc.conf"}
    };
    int argc = sizeof(argv);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      return 1;
    }


    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2);

    feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_opts);
    if (feature_info_->feature_type != "mfcc") {
        KALDI_ERR << "feature_type should be mfcc";
    }
   
    samp_freq_ = feature_info_->mfcc_opts.frame_opts.samp_freq; 
    chunk_length_ = samp_freq_ * chunk_length_secs;
        

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
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts,
                                                                &am_nnet_);
    decode_fst_ = ReadFstKaldiGeneric(fst_rxfilename);

    if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename))) {
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
    }

    adaption_state_ = new OnlineIvectorExtractorAdaptationState(
                          feature_info.ivector_extractor_info);
    

}

Decoder* DecoderFactory::StartDecodingSession() {
    return new Decoder(trans_model_, *decode_fst_, ...);
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
DecoderFactory* InitDecoderFactory(const char* resource_dir) {
    return new DecoderFactory(resource_dir);
}

// Creates a decoder object.
// Returns nullptr on failure.
Decoder* StartDecodingSession(const DecoderFactory* decoder_factory) {
    return decoder_factory->StartDecodingSession();
}
// Feed PCM SI16 data into the decoder.
// Returns 0 on success, error code otherwise.
int32_t FeedChunk(Decoder* decoder, uint16_t *data, size_t length){

    return decoder->FeedChunk(data,length);

}
/*
Gets current (ongoing) recognition result,
probably as a JSON or maybe protobuf
(with word timings and other stuff).
*/
std::string GetCurrentResult(const Decoder*);
// Frees the resources and destroys the recognition session.
// Returns 0 on success, error code otherwise.
int32_t Finalize(Decoder*);

