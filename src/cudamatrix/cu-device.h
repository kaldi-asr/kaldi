#ifndef KALDI_CUDAMATRIX_CUDEVICE_H_
#define KALDI_CUDAMATRIX_CUDEVICE_H_

#if HAVE_CUDA==1

#include <map>
#include <string>
#include <iostream>

namespace kaldi {

/**
 * Singleton object which represents CUDA device
 * responsible for CUBLAS initilalisation, collects profiling info
 */
class CuDevice {
 // Singleton interface...
 private:
  CuDevice();
  CuDevice(CuDevice&);
  CuDevice& operator=(CuDevice&);

 public:
  ~CuDevice();
  static CuDevice& Instantiate() { 
    return msDevice; 
  }

 private:
  static CuDevice msDevice;


 /**********************************/
 // Instance interface
 public:
 
  /// Check if the CUDA device is in the system      
  bool Enabled() { 
    return enabled_; 
  }

  void Verbose(bool verbose) { 
    verbose_ = verbose; 
  }

  /// Sum the IO time
  void AccuProfile(const std::string& key,double time);
  void PrintProfile(); 

  void ResetProfile() { 
    profile_map_.clear(); 
  }

  std::string GetFreeMemory();

 private:
  std::map<std::string, double> profile_map_;
  bool enabled_;
  bool verbose_;

}; //class CuDevice


}//namespace

#endif //HAVE_CUDA


#endif
