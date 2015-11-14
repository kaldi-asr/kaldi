namespace kaldi {
namespace segmenter {

class SpeechActivityDetector {
 public:
   /** 
    * Speech activity methods are described as follows:
    * kAvgMaxThreshold : Look at the average and the max in the 100 frame 
    * window. For speech frames, in the 100 frame i.e. 1s window, the regions
    * of high energy will be intertwined with regions of silence. So there would
    * definitely be peaks that are above the average in that window. To decide
    * on a particular frame, we check if its energy is above 
    * alpha * avg_energy + beta * max_energy
                               
   **/ 
   enum SadMethod {
    kAvgMaxThreshold,     
    kEnergyThreshold
   };

 private:

   // The method of speech activity detection to be used. This method 
   // determines the type of input required.
   int32 sad_method;
};

}
}
