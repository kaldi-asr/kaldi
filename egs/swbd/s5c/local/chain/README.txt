
there are a lot of tuning experiments here.

ones to look at right now:
  2y is a TDNN baseline
  4f is a good jesus-layer system
  4q is an improved TDNN with various bells and whistles from Vijay.
  4r is a slightly-better jesus-layer system than 4f, with one more layer.
  5e is the best configuration run so far that doesn't have statistics-averaging layers.
  5g uses a statistics-averaging layer in the middle to slightly improve on 5e (by about
     0.2%).
  5j is a basic configuration without iVectors (about 2% abs worse than 5e)
  5k is the best configurations without iVectors... about 1% abs worse than 5e; we
     use statistics-averaging layers to do some crude adaptation.
  5t gives about the same performance as 5e but is about 30% faster to train
     and is smaller.
  5v is what I am currently using as a baseline- it has an even smaller
     --jesus-hidden-dim as 5t (hence faster to train), but gives the same
     performance.
  6g is a setup with a 'thinner' jesus-layer (with only one repeated-affine component)
     and slightly more parameters, which is quicker to train than 5v but gives
     about the same results.  I'm hoping to use this setup, going forward.
  6i is like 6i but with a separate last-but-one affine layer for the xent output
     (marginally better than 6g).
  6z is probably the thing I currently recommend to run-- it's a TDNN+ReLU based
     setup that's quite fast to train and gives better results than our old
     jesus-layer-based system.


