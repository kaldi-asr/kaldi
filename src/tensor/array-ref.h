/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {


// Similar to llvm/PyTorch's ArrayRef, this is a lightweight way to store an
// array (zero or more elements of type T).  The array is not owned here; it
// will generally be unsafe to use an ArrayRef as other than a local variable.
//
// ArrayRef has only two members and it will probably make sense to pass it by
// value most of the time.
template <typename T>
struct ArrayRef {
  // Note:
  uint64_t size;
  const int64_t *data;

  // Lots to do here.

  inline int64_t operator (uint64_t i) const {
    KALDI_ASSERT(i < size);
    return data[i];
  }

  // cast to std::vector; for cases where you might need to
  // change the contents or keep it more permanently.
  operator std::vector<int64_t> () const;
};




};
