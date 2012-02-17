
#if HAVE_CUDA==1

#include <cublas.h>
#include <cuda.h>

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-error.h"


namespace kaldi {

CuDevice::CuDevice()
 : enabled_(false), verbose_(true) {
  int ret;
  if((ret = cublasInit()) == 0) {
    enabled_ = true;
  } else {
    //WHY IS THE STRING STATIC IN 
    //base/kaldi-error.cc:GetProgramName?
    //KALDI_WARN CAUSES SEGMENTATION FAULT!!!
    //KALDI_WARN << "CUDA will not be used!!! cublasInit() returns: " << ret;
    std::cerr << "WARNING (CUDA will not be used!!! cublasInit() returns: " << ret << ")\n";
  }
}


CuDevice::~CuDevice() {
  if(enabled_) {
    cuSafeCall(cublasShutdown());
  } else {
    //WHY IS THE STRING STATIC IN 
    //base/kaldi-error.cc:GetProgramName?
    //KALDI_WARN CAUSES SEGMENTATION FAULT!!!
    //KALDI_WARN << "CUDA was not used";
    std::cerr << "WARNING (CUDA was not used)\n";
  }
}


void CuDevice::AccuProfile(const std::string& key,double time) { 
  if(profile_map_.find(key) == profile_map_.end()) {
    profile_map_[key] = 0.0;
  }
  profile_map_[key] += time;
}


void CuDevice::PrintProfile() {
  if(verbose_ && enabled_) { 
    std::ostringstream os;
    os << "-----\n[cudevice profile]\n";
    std::map<std::string, double>::iterator it;
    for(it = profile_map_.begin(); it != profile_map_.end(); ++it) {
      os << it->first << "\t" << it->second << "s\n";
    }
    os << "-----";
    KALDI_LOG << os.str();
  }
}


std::string CuDevice::GetFreeMemory() {
  size_t mem_free, mem_total;
  cuMemGetInfo(&mem_free, &mem_total);
  std::ostringstream os;
  os << "Free:" << mem_free/(1024*1024) << "MB "
     << "Used:" << (mem_total-mem_free)/(1024*1024) << "MB "
     << "Total:" << mem_total/(1024*1024) << "MB";
  return os.str();
}



////////////////////////////////////////////////
// The instance of the static singleton 
//
CuDevice CuDevice::msDevice;
//
////////////////////////////////////////////////



}


#endif //HAVE_CUDA
