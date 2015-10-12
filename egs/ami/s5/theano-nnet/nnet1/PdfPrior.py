import sys, os, numpy as np

class PdfPrior():

  log_priors_ = ""
  prior_scale_ = 1.0
  prior_floor_ = 1e-10
  
  def __init__(self, opts):
    
    if opts.class_frame_counts == "":
      # class_frame_counts is empty, the PdfPrior is deactivated...
      # (for example when 'nnet-forward' generates bottleneck features)
      return

    sys.stderr.write("Computing pdf-priors from : %s" %opts.class_frame_counts)
    # list of frame counts
    frame_counts = map( lambda x: int(x), open(opts.class_frame_counts, "r").readline().rstrip().split()[1:-1])

    rel_freq = np.asarray(frame_counts, dtype='float32')
    rel_freq = rel_freq/np.sum(rel_freq)

    # get log-priors
    self.log_priors_ = rel_freq + 1e-20
    self.log_priors_ = np.log(self.log_priors_)

    sqrt_FLT_MAX = 1.84467e+19 #C++ Float Max
    num_floored = 0
    for i in xrange(self.log_priors_.shape[0]):
      if rel_freq[i] < self.prior_floor_:
        self.log_priors_[i] = sqrt_FLT_MAX
        num_floored = num_floored+1
        
    # print "Floored " + str(num_floored) + " pdf-priors"
    # print "(hard-set to "+ str(np.sqrt(FLT_MAX)) + ", which disables DNN output when decoding)";
    
    sys.stderr.write("Floored %d pdf-priors \n" %num_floored);
    sys.stderr.write("hard-set to %f, which disables DNN output when decoding \n" %sqrt_FLT_MAX);


  def SubtractOnLogpost(self, llk):
    
    llk = llk + (-self.prior_scale_) * self.log_priors_
    return llk

