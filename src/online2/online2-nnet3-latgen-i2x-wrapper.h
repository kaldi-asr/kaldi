// online2/online2-nnet3-latgen-i2x-wrapper.h
#ifndef KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_
#define KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_
#include <cstdint>

namespace kaldi {
class DecoderFactoryImpl;
class DecoderImpl;
} // namespace kaldi

using kaldi::DecoderFactoryImpl;
using kaldi::DecoderImpl;

struct RecognitionResult {
  std::string transcript;
};

class Decoder {
 public:
  ~Decoder();

  // Feed PCM SI16 data into the decoder.
  // Returns 0 on success, error code otherwise.
  // If called with length == 0, the decoder is finalized and no further calls are allowed.
  int32_t FeedBytestring(const std::string& bytestring);

  // Puts the final recognition result in the string passed by pointer.
  // Frees the resources and destroys the recognition session.
  // Returns 0 on success, error code otherwise.
  int32_t GetResultAndFinalize(RecognitionResult *recognition_result);
 private:
  DecoderImpl *decoder_impl_ = nullptr;
  Decoder(DecoderImpl *decoder_impl_);
  friend class DecoderFactory;

  Decoder& operator=(const Decoder&) = delete;
  Decoder(const Decoder&) = delete;
};

class DecoderFactory {
 public:
  /*
   Creates a decoder factory.
   Called only once during the lifetime.
   mmaps extremely heavy resources like WFST and AM (up to 20 GB).
   Calling it more than once is likely to cause OOM!
   Returns a handle to a decoder factory,
   which will create light-weighted decoder objects (one per session).
  */
  explicit DecoderFactory(const std::string &resource_dir);
  ~DecoderFactory(); // TODO switch to a singleton for DecoderFactory

  // Creates a decoder object.
  // Returns nullptr on failure.
  Decoder *StartDecodingSession();
 private:
  DecoderFactoryImpl *decoder_factory_impl_ = nullptr;

  DecoderFactory& operator=(const DecoderFactory&) = delete;
  DecoderFactory(const DecoderFactory&) = delete;
};

#endif // KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_

