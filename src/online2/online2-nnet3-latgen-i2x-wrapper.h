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
  bool is_final = false;
  bool error = false;
};

class Decoder {
 public:
  ~Decoder();

  // Feed PCM SI16 data into the decoder.
  // Returns 0 on success, error code otherwise.
  int32_t FeedBytestring(const std::string& bytestring);

  // Finalize the decoder, signaling no more data will be available.
  // Does not free the resources.
  // Further calls to FeedBytestring will be ignored and return error code.
  int32_t Finalize();

  // Return recognition result.
  // Will return partial result if the decoder is not yet finalized.
  const RecognitionResult GetResult();

  // Frees the internal memory structures.
  // All further calls to this Decoder object will be ignored and return error code.
  void InvalidateAndFree();
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
  // NOTE: PASSES OWNERSHIP TO THE CALLER.
  // Returns nullptr on failure.
  Decoder *StartDecodingSession();
 private:
  DecoderFactoryImpl *decoder_factory_impl_ = nullptr;

  DecoderFactory& operator=(const DecoderFactory&) = delete;
  DecoderFactory(const DecoderFactory&) = delete;
};

#endif // KALDI_ONLINE2_ONLINE_NNET3_LATGEN_I2X_WRAPPER_H_

