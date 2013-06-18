Intro
-----
The repository contains the first attempt to build
online Kaldi decoder taking raw audio packets.

The decoder should have simpel interface,
because the next step will be interfacing the functionality of the decoder from Python.

Workflow of KALDI decoding in few lines
---------------------
```cpp
Classes:
    OnlineFeInput<Source, Mffc>
    MfccOptions // Usage: Mfcc mfcc(mfcc_opts);
    Mfcc   - ?place holder for mfcc features?

    Mfcc mfcc(mfcc_opts);
    FeInput fe_input(&au_src, &mfcc, ..)
    OnlineCmvnInput cmvn_input(&fe_input, ..);
    feat_transform = new OnlineLdaInput(&cmvn_input, ..)
    OnlineDecodableDiagGmmScaled decodable(feat_transform, ..)
    while (1) {
      OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);
      // different staff for online decoder decoder.FinishTraceBack(&out_fst);
```


Classes for online decoder
--------------------------
```cpp
// in online/online-audio-source.h
class OnlineVectorSource 
    int32 Read(VectorBase<BaseFloat> *data, uint32 *timeout = 0);

// in online/online-decodable.h
// A decodable, taking input from an OnlineFeatureInput object on-demand
class OnlineDecodableDiagGmmScaled : public DecodableInterface 
  virtual BaseFloat LogLikelihood(int32 frame, int32 index);
  virtual bool IsLastFrame(int32 frame);
  virtual int32 NumIndices() /// Indices are one-based!  This is for compatibility with OpenFst.

// in online/online-fast-input.h
class OnlineFeatInputItf
  virtual bool Compute(Matrix<BaseFloat> *output, uint32 *timeout) = 0;

// in online/online-cmn.h
class OnlineCMN 
    ApplyCmvn

// in online/online-faster-decoder.h
struct OnlineFasterDecoderOpts : public FasterDecoderOptions
  void Register(ParseOptions *po, bool full)
    

// in online/online-faster-decoder.h
class OnlineFasterDecoder : public FasterDecoder 
  // Codes returned by Decode() to show the current state of the decoder
  enum DecodeState {
    kEndFeats = 1, // No more scores are available from the Decodable
    kEndUtt = 2, // End of utterance, caused by e.g. a sufficiently long silence
    kEndBatch = 4 // End of batch - end of utterance not reached yet
  };
  DecodeState Decode(DecodableInterface *decodable);

  // Makes a linear graph, by tracing back from the last "immortal" token
  // to the previous one
  bool PartialTraceback(fst::MutableFst<LatticeArc> *out_fst);

  // Makes a linear graph, by tracing back from the best currently active token
  // to the last immortal token. This method is meant to be invoked at the end
  // of an utterance in order to get the last chunk of the hypothesis
  void FinishTraceBack(fst::MutableFst<LatticeArc> *fst_out);

  // Returns "true" if the best current hypothesis ends with long enough silence
  bool EndOfUtterance();

  int32 frame() { return frame_; }

```


Reading the wav source
----------------------
```cpp
    SequentialTableReader<WaveHolder> reader(std::string wav_rspecifier);
```
