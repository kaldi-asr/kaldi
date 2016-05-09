// TODO: License header

#if HAVE_CUDNN == 1
#include <cudnn.h>

namespace kaldi {
  namespace cudnn {

    // is_same() is defined in the the ++ standard library, but only in C+11 
    // and onward. We need these because we cannot assume that users have a 
    // C++11 compiler.
    template<class T, class U>
      struct is_same {
	enum { value = 0 };
      };

    template<class T>
      struct is_same<T, T> {
      enum { value = 1 };
    };

    inline cudnnDataType_t GetDataType() {
      if (is_same<BaseFloat, float>::value)
	return CUDNN_DATA_FLOAT;
      else if (is_same<BaseFloat, double>::value)
	return CUDNN_DATA_DOUBLE;
      else {
	KALDI_ERR << "Unsupported type.";
	return CUDNN_DATA_FLOAT;
      }
    }

    const BaseFloat one  = 1;
    const BaseFloat zero = 0;
  } // end namespace cudnn
} // end namespace kaldi
#endif
