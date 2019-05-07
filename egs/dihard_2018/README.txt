 
 This is a Kaldi recipe for The First DIHARD Speech Diarization Challenge.  
 DIHARD is a new annual challenge focusing on "hard" diarization; that is,
 speech diarization for challenging corpora where there is an expectation that
 the current state-of-the-art will fare poorly, including, but not limited
 to: clinical interviews, extended child language acquisition recordings,
 YouTube videos and "speech in the wild" (e.g., recordings in restaurants)
 See https://coml.lscp.ens.fr/dihard/index.html for details.

 The subdirectories "v1" and so on are different speaker diarization
 recipes. The recipe in v1 demonstrates a standard approach using a
 full-covariance GMM-UBM, i-vectors, PLDA scoring and agglomerative
 hierarchical clustering. The example in v2 demonstrates DNN speaker 
 embeddings, PLDA scoring and agglomerative hierarchical clustering.
